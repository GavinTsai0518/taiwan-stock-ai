"""
ml_trend_model.py

真正的機器學習趨勢預測模型（短期 4 個交易日 + 中期 15 個交易日雙週期），
用來取代/驗證 paper_trading.py 目前那套規則式多因子評分（cross_market_validation.py
測出那套規則手調公式在美日個股上幾乎沒有預測力，相關係數趨近於 0）。

修正舊版 ML 模型（本專案更早之前拆掉的 XGBoost/LightGBM/RandomForest 集成）的兩個致命 bug：
1. 訓練/預測資料重疊：舊版對 model.fit() 用過的訓練集自己的最後一列做 predict_proba，
   等於模型在背答案。這裡改成：訓練集只用「已經有明確標籤」的歷史列（最近 N 天因為
   還沒發生無法算出標籤，天生就不會進訓練集），預測永遠只對「今天」這一列做，並用
   assert 在執行期驗證兩者的 index 不重疊。
2. 未來函數：舊版誤判月營收公告日期，讓模型在資料實際公開前就看得到未來的營收數字。
   這裡的 merge_revenue_features() 明確用「次月10號前公告」的規則，逐日判斷當天真正
   看得到的是哪一期營收。

這支檔案本身不是每日排程的一部分，是被 ml_trend_validation.py 呼叫做 walk-forward 回測用的
核心邏輯（特徵工程、標籤產生、訓練、預測）。重用 paper_trading.py 已經寫好且驗證過的資料源
函式（TWSE T86 法人資料、MOPS 月營收、FinMind 股價），不重新發明資料擷取邏輯。
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from lightgbm import LGBMClassifier

from paper_trading import dl, get_atr14, fetch_twse_institutional_day, _fetch_mops_revenue_month

# ===== 標籤參數：短期用原本三重屏障的倍數，中期拉長窗口、放寬 ATR 倍數避免雜訊蓋過訊號 =====
SHORT_WINDOW, SHORT_TP_ATR, SHORT_SL_ATR = 4, 1.8, 1.0
MEDIUM_WINDOW, MEDIUM_TP_ATR, MEDIUM_SL_ATR = 15, 3.5, 2.0

FEATURE_COLUMNS = [
    'ma5_ma20_ratio', 'ma20_ma60_ratio', 'bias_20', 'rsi14', 'ret_5d', 'ret_20d', 'vol_ratio',
    'dist_to_support_20d', 'dist_to_resistance_20d', 'support_touch_count_20d',
    'dist_to_support_60d', 'dist_to_resistance_60d', 'support_touch_count_60d',
    'foreign_positive_days_10d', 'trust_positive_days_10d', 'chip_strength_5d',
    'revenue_yoy',
    'mkt_ma20_ma60_ratio', 'mkt_volatility',
]

# ==========================================
# 特徵工程
# ==========================================
def compute_technical_features(df):
    """技術面特徵：保留原始數值（不像 paper_trading.score_technical 那樣先壓成 0-100 分），
    讓模型自己從資料中學非線性關係，而不是先幫模型下結論。"""
    close = df['Close']
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    df['ma5_ma20_ratio'] = (ma5 - ma20) / ma20 * 100
    df['ma20_ma60_ratio'] = (ma20 - ma60) / ma60 * 100
    df['bias_20'] = (close - ma20) / ma20 * 100

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi14'] = 100 - (100 / (1 + gain / (loss + 1e-6)))

    df['ret_5d'] = close.pct_change(5) * 100
    df['ret_20d'] = close.pct_change(20) * 100

    vol_ma20 = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / (vol_ma20 + 1e-6)
    return df

def compute_support_resistance_features(df, windows=(20, 60)):
    """圖形支撐/壓力特徵：距離近期高低點的百分比 + 該價位過去被測試的次數
    （次數越多代表這個價位被反覆驗證過，支撐/壓力的意義越強）。
    這裡只提供明確可計算的數值特徵，不預先幫模型下結論（例如不寫死「跌破支撐扣分」），
    由模型在訓練中自己學到這些特徵跟未來報酬的關係，符合「讓 AI 自己發現訊號」的目的。"""
    close = df['Close']
    for w in windows:
        rolling_low = df['Low'].rolling(w).min()
        rolling_high = df['High'].rolling(w).max()
        df[f'dist_to_support_{w}d'] = (close - rolling_low) / close * 100
        df[f'dist_to_resistance_{w}d'] = (rolling_high - close) / close * 100
        near_support = (close - rolling_low).abs() / close < 0.01
        df[f'support_touch_count_{w}d'] = near_support.rolling(w).sum()
    return df

def build_chip_dataframe(inst_data_by_date, stock_id):
    """把 {日期: {股票代碼: {...}}} 轉成單一股票的法人買賣超時間序列 DataFrame。"""
    rows = []
    for date_str, day_map in inst_data_by_date.items():
        rec = day_map.get(stock_id)
        rows.append({
            'date': pd.to_datetime(date_str, format='%Y%m%d'),
            'foreign_net': rec['foreign'] if rec else 0,
            'trust_net': rec['trust'] if rec else 0,
        })
    if not rows:
        return pd.DataFrame(columns=['date', 'foreign_net', 'trust_net'])
    return pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

def merge_chip_features(df, chip_df):
    """籌碼面特徵：用「近N日正買超天數」代替嚴格的連續 streak（後者不易向量化，
    在對整段歷史逐日計算特徵時，用滾動計數是同樣有資訊量但正確、快速的做法）。"""
    df = df.merge(chip_df, on='date', how='left')
    df[['foreign_net', 'trust_net']] = df[['foreign_net', 'trust_net']].fillna(0)
    combined_5d = (df['foreign_net'] + df['trust_net']).rolling(5).sum()
    volume_5d = df['Volume'].rolling(5).sum()
    df['chip_strength_5d'] = combined_5d / (volume_5d + 1e-6)
    df['foreign_positive_days_10d'] = (df['foreign_net'] > 0).rolling(10).sum()
    df['trust_positive_days_10d'] = (df['trust_net'] > 0).rolling(10).sum()
    return df

def merge_revenue_features(df, revenue_by_month, stock_id):
    """基本面特徵：逐日判斷「當天實際看得到」的最新一期營收 YoY，明確套用「次月10號前公告」
    的規則，避免重蹈舊版 ML 模型的未來函數 bug（讓模型在資料公開前就看到未來數字）。
    revenue_by_month 的 key 是「營收所屬月份」(year, month)，不是公告月份。"""
    def _known_yoy(row_date):
        y, m = row_date.year, row_date.month
        if row_date.day >= 10:
            rev_y, rev_m = (y, m - 1) if m > 1 else (y - 1, 12)
        else:
            rev_y, rev_m = y, m - 2
            if rev_m <= 0:
                rev_m += 12
                rev_y -= 1
        rec = revenue_by_month.get((rev_y, rev_m), {}).get(stock_id)
        return rec['yoy_pct'] if rec else np.nan

    df['revenue_yoy'] = df['date'].apply(_known_yoy)
    return df

def fetch_market_context_features(lookback_days=900):
    """大盤環境特徵（0050 代理），用連續數值而非類別文字（BULL/BEAR/NORMAL），
    直接餵給模型比較方便，讓模型自己學市場環境跟個股表現的交互關係。"""
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    df_market = dl.taiwan_stock_daily(stock_id='0050', start_date=start_date)
    if df_market.empty:
        return pd.DataFrame(columns=['date', 'mkt_ma20_ma60_ratio', 'mkt_volatility'])
    df_market = df_market.rename(columns={'close': 'Close'})
    df_market['date'] = pd.to_datetime(df_market['date'])
    df_market = df_market.sort_values('date').reset_index(drop=True)
    ma20 = df_market['Close'].rolling(20).mean()
    ma60 = df_market['Close'].rolling(60).mean()
    df_market['mkt_ma20_ma60_ratio'] = (ma20 - ma60) / ma60 * 100
    ret = df_market['Close'].pct_change()
    df_market['mkt_volatility'] = ret.rolling(20).std() * 100
    return df_market[['date', 'mkt_ma20_ma60_ratio', 'mkt_volatility']]

# ==========================================
# 標籤（三重屏障法：短期/中期用不同窗口與 ATR 倍數）
# ==========================================
def compute_barrier_labels(df, window, tp_atr, sl_atr):
    atr = get_atr14(df)
    labels = []
    for i in range(len(df)):
        if i + window >= len(df):
            labels.append(np.nan)  # 未來資料不足，無法判定，之後會被排除在訓練集外
            continue
        entry = df['Close'].iloc[i]
        cur_atr = atr.iloc[i]
        if pd.isna(cur_atr) or cur_atr <= 0:
            cur_atr = entry * 0.02
        upper = entry + tp_atr * cur_atr
        lower = entry - sl_atr * cur_atr
        future = df.iloc[i + 1: i + 1 + window]
        hit = 0
        for _, row in future.iterrows():
            if row['High'] >= upper:
                hit = 1
                break
            if row['Low'] <= lower:
                hit = 0
                break
        labels.append(hit)
    return pd.Series(labels, index=df.index)

# ==========================================
# 歷史資料蒐集（訓練用，回溯抓取，重用 paper_trading.py 已驗證過的資料源函式）
# ==========================================
def fetch_historical_institutional_data(trading_days_back=500, max_lookback_days=None):
    """回溯抓取全市場三大法人資料（TWSE T86，逐日呼叫，無 FinMind 額度限制）。"""
    if max_lookback_days is None:
        max_lookback_days = int(trading_days_back * 1.6) + 30
    results = {}
    d = datetime.now()
    tries = 0
    while len(results) < trading_days_back and tries < max_lookback_days:
        date_str = d.strftime('%Y%m%d')
        day_data = fetch_twse_institutional_day(date_str)
        if day_data:
            results[date_str] = day_data
        d -= timedelta(days=1)
        tries += 1
        time.sleep(0.25)
    return results

def fetch_historical_revenue_data(months_back=24):
    """回溯抓取全市場月營收（MOPS，逐月呼叫）。key 是「營收所屬月份」，不是公告月份。"""
    d = datetime.now().replace(day=1) - timedelta(days=1)
    y, m = d.year, d.month
    results = {}
    for _ in range(months_back):
        data = _fetch_mops_revenue_month(y - 1911, m)
        if data:
            results[(y, m)] = data
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
        time.sleep(0.3)
    return results

# ==========================================
# 組裝單一股票的完整訓練用 DataFrame
# ==========================================
def build_training_dataframe(stock_id, price_start_date, inst_data_by_date, revenue_by_month, market_ctx_df):
    df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=price_start_date)
    if df.empty or len(df) < 100:
        return None
    df = df.rename(columns={'max': 'High', 'min': 'Low', 'close': 'Close',
                             'open': 'Open', 'Trading_Volume': 'Volume'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df = compute_technical_features(df)
    df = compute_support_resistance_features(df)

    chip_df = build_chip_dataframe(inst_data_by_date, stock_id)
    df = merge_chip_features(df, chip_df)

    df = merge_revenue_features(df, revenue_by_month, stock_id)

    if not market_ctx_df.empty:
        df = df.merge(market_ctx_df, on='date', how='left')
    else:
        df['mkt_ma20_ma60_ratio'] = np.nan
        df['mkt_volatility'] = np.nan

    df['label_short'] = compute_barrier_labels(df, SHORT_WINDOW, SHORT_TP_ATR, SHORT_SL_ATR)
    df['label_medium'] = compute_barrier_labels(df, MEDIUM_WINDOW, MEDIUM_TP_ATR, MEDIUM_SL_ATR)

    return df

# ==========================================
# 訓練 + 預測（正確的樣本外預測，含執行期資料洩漏檢查）
# ==========================================
def train_and_predict(df, horizon='short'):
    """回傳 (win_prob, model, X_train)。win_prob 是對「今天」這一列做的樣本外預測機率；
    資料不足或今天特徵有缺值時回傳 (None, None, None)。

    正確性保證：訓練集 X_train 只包含 label 非 NaN 的歷史列，而「今天」這一列的 label
    必為 NaN（因為未來還沒發生，三重屏障法算不出結果），兩者的 index 天生不重疊，
    並用 assert 在執行期驗證，一旦未來改動不小心破壞這個保證會立刻噴錯，而不是默默錯下去。"""
    label_col = 'label_short' if horizon == 'short' else 'label_medium'

    df_clean = df.dropna(subset=FEATURE_COLUMNS + [label_col])
    if len(df_clean) < 60 or df_clean[label_col].nunique() < 2:
        return None, None, None

    X_train = df_clean[FEATURE_COLUMNS]
    y_train = df_clean[label_col]

    model = LGBMClassifier(n_estimators=60, learning_rate=0.03, max_depth=4, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    today_row = df.iloc[[-1]]
    if today_row[FEATURE_COLUMNS].isna().any(axis=1).iloc[0]:
        return None, model, X_train
    X_today = today_row[FEATURE_COLUMNS]

    assert X_today.index[0] not in X_train.index, "資料洩漏：今天這列不應該出現在訓練集裡！"

    win_prob = float(model.predict_proba(X_today)[0][1])
    return win_prob, model, X_train
