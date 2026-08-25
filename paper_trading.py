import pandas as pd
import numpy as np
import sqlite3
import warnings
import time
import os
import sys
from datetime import datetime, timedelta
from FinMind.data import DataLoader

warnings.filterwarnings('ignore')

DB_NAME = "paper_trading.db"

# ===== 可調參數（依「AI 選股引擎規格書」第 9 節建議預設值，之後可依實戰結果調整）=====
WEIGHT_TECH = 0.4
WEIGHT_FUND = 0.3
WEIGHT_CHIP = 0.3
ENTRY_SCORE_THRESHOLD = 70      # 一般/多頭體制進場總分門檻
BEAR_SCORE_THRESHOLD = 80       # 空頭體制進場總分門檻（提高）
MIN_FACTOR_SCORE = 40           # 三因子單項一票否決門檻
K_SL = 1.75                     # 停損 ATR 倍數（規格建議 1.5~2.0，取中間值）
K_TP = 3.0                      # 停利 ATR 倍數
RISK_PER_TRADE_PCT = 1.0        # 單筆風險占總資金 %（空頭體制會減半）
MAX_POSITION_PCT = 10.0         # 單檔部位上限 %
MARKET_PROXY_ID = '0050'        # FinMind 的 taiwan_stock_daily 沒有加權指數代碼，改用追蹤大盤的 0050 ETF 代理

# FinMind Token 一律從環境變數讀取（GitHub Actions 用 repo secret 注入），不再寫死於程式碼中
dl = DataLoader()
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN")

if not FINMIND_TOKEN:
    print("❌ 未設定 FINMIND_TOKEN 環境變數，無法登入 FinMind，中止執行。")
    sys.exit(1)

try:
    dl.login_by_token(api_token=FINMIND_TOKEN)
    print("✅ FinMind API Token 驗證成功！高流量存取功能已啟動。")
except Exception as e:
    print(f"❌ API Token 登入失敗: {e}")
    sys.exit(1)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            predict_date TEXT,
            stock_id TEXT,
            stock_name TEXT,
            latest_price REAL,
            buy_price REAL,
            tp_price REAL,
            sl_price REAL,
            ai_win_rate REAL,
            status TEXT DEFAULT 'PENDING',
            real_max_price REAL DEFAULT 0,
            real_min_price REAL DEFAULT 0,
            validated_date TEXT,
            revenue_yoy REAL,
            pe_ratio REAL,
            position_size REAL DEFAULT 0.0,
            market_regime TEXT DEFAULT 'NORMAL',
            trailing_stop_price REAL,
            entry_atr REAL
        )
    ''')
    # 實際上線的 predictions 表是專案最早上傳時的舊 schema（只有 12 欄），
    # CREATE TABLE IF NOT EXISTS 對已存在的表是 no-op，所以這些欄位一直沒被補上，
    # 導致任何一次 INSERT 只要用到這些欄位就會噴 "no such column" 並被外層 except 吞掉。
    # 用 ALTER TABLE 逐一補齊，欄位已存在時忽略錯誤。
    for col_sql in ("ALTER TABLE predictions ADD COLUMN revenue_yoy REAL",
                     "ALTER TABLE predictions ADD COLUMN pe_ratio REAL",
                     "ALTER TABLE predictions ADD COLUMN position_size REAL DEFAULT 0.0",
                     "ALTER TABLE predictions ADD COLUMN market_regime TEXT DEFAULT 'NORMAL'",
                     "ALTER TABLE predictions ADD COLUMN trailing_stop_price REAL",
                     "ALTER TABLE predictions ADD COLUMN entry_atr REAL"):
        try:
            cursor.execute(col_sql)
        except sqlite3.OperationalError:
            pass

    # 每日觀察報告：不論是否達進場門檻，都記錄總分前十名 + 成交量異常放大名單，每支股票附個別報告文字
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT,
            stock_id TEXT,
            stock_name TEXT,
            category TEXT,
            rank INTEGER,
            latest_price REAL,
            total_score REAL,
            tech_score REAL,
            fund_score REAL,
            chip_score REAL,
            volume REAL,
            volume_avg20 REAL,
            volume_surge_pct REAL,
            revenue_yoy REAL,
            pe_ratio REAL,
            report_text TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_atr14(df):
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(14).mean()

# ==========================================
# 訊號失效判斷（技術面死亡交叉 / 籌碼面連續賣超），用於「依訊號動態出場」
# ==========================================
def check_signal_invalid(stock_id):
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    try:
        df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
        time.sleep(0.1)
    except Exception:
        return False, ""

    if df.empty or len(df) < 25:
        return False, ""

    df = df.rename(columns={'close': 'Close'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    ma5 = df['Close'].rolling(5).mean()
    ma20 = df['Close'].rolling(20).mean()
    if ma5.notna().sum() >= 2 and ma20.notna().sum() >= 2:
        if ma5.iloc[-2] >= ma20.iloc[-2] and ma5.iloc[-1] < ma20.iloc[-1]:
            return True, "訊號轉空出場 (MA5 死亡交叉跌破 MA20)"

    try:
        df_chip = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
        time.sleep(0.1)
        if not df_chip.empty:
            foreign_net = df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['buy'].sum() - \
                          df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['sell'].sum()
            trust_net = df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['buy'].sum() - \
                        df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['sell'].sum()
            combined = foreign_net.add(trust_net, fill_value=0).sort_index()
            last3 = combined.tail(3)
            if len(last3) == 3 and (last3 < 0).all():
                return True, "訊號轉空出場 (三大法人連續 3 日賣超)"
    except Exception:
        pass

    return False, ""

# ==========================================
# 結算與追蹤：ATR 移動停利 + 訊號失效出場（操作週期不固定，依訊號動態調整）
# ==========================================
def audit_past_predictions():
    init_db()
    conn = sqlite3.connect(DB_NAME)
    pending_df = pd.read_sql("SELECT * FROM predictions WHERE status='PENDING'", conn)

    if pending_df.empty:
        conn.close()
        return

    today_str = datetime.now().strftime('%Y-%m-%d')
    cursor = conn.cursor()

    for _, row in pending_df.iterrows():
        p_id = row['id']
        stock_id = row['stock_id']
        p_date = row['predict_date']
        buy_price = row['buy_price']
        tp_price = row['tp_price']
        original_sl = row['sl_price']

        entry_atr = row.get('entry_atr')
        if pd.isna(entry_atr) or not entry_atr:
            # 舊資料沒存 entry_atr，用 K_TP 反推估計（僅適用於 K_TP 常數未被調整過的舊倉位）
            entry_atr = (tp_price - buy_price) / K_TP if (tp_price and buy_price) else None

        current_stop = row.get('trailing_stop_price')
        if pd.isna(current_stop) or not current_stop:
            current_stop = original_sl

        try:
            df_real = dl.taiwan_stock_daily(stock_id=stock_id, start_date=p_date)
            time.sleep(0.1)
        except Exception:
            continue

        if df_real.empty or 'date' not in df_real.columns:
            continue

        df_future = df_real[df_real['date'] > p_date]
        if df_future.empty:
            continue

        running_high = df_future['max'].max()
        running_low = df_future['min'].min()

        # ATR 移動停利：漲超過 1 倍 ATR 先保本，之後每漲 1 倍 ATR 停損同步上移 1 倍 ATR
        if entry_atr and entry_atr > 0:
            n = int((running_high - buy_price) // entry_atr)
            if n >= 1:
                candidate_stop = buy_price + (n - 1) * entry_atr
                current_stop = max(current_stop, candidate_stop, original_sl)

        status = 'PENDING'
        if running_high >= tp_price:
            status = 'WIN (成功停利)'
        elif running_low <= current_stop:
            status = 'LOSS (觸及停損)' if current_stop <= original_sl + 1e-6 else 'WIN (移動停利出場)'
        else:
            invalid, reason = check_signal_invalid(stock_id)
            if invalid:
                last_close = df_future['close'].iloc[-1] if 'close' in df_future.columns else None
                if last_close is None:
                    status = 'EXPIRED (訊號出場)'
                elif last_close > buy_price:
                    status = 'WIN (訊號出場)'
                elif last_close < buy_price:
                    status = 'LOSS (訊號出場)'
                else:
                    status = 'EXPIRED (訊號出場)'

        if status != 'PENDING':
            cursor.execute('''
                UPDATE predictions
                SET status=?, real_max_price=?, real_min_price=?, validated_date=?, trailing_stop_price=?
                WHERE id=?
            ''', (status, running_high, running_low, today_str, current_stop, p_id))
        else:
            cursor.execute('UPDATE predictions SET trailing_stop_price=? WHERE id=?', (current_stop, p_id))

    conn.commit()
    conn.close()

def detect_market_regime():
    start_date = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
    try:
        df_market = dl.taiwan_stock_daily(stock_id=MARKET_PROXY_ID, start_date=start_date)
        time.sleep(0.1)
        if df_market.empty or len(df_market) < 30:
            return 'NORMAL', 0.0

        df_market = df_market.rename(columns={'close': 'Close'})
        sma_20 = df_market['Close'].rolling(20).mean().iloc[-1]
        sma_60 = df_market['Close'].rolling(60).mean().iloc[-1] if len(df_market) >= 60 else sma_20
        latest_close = df_market['Close'].iloc[-1]

        if latest_close < sma_20 and sma_20 < sma_60:
            regime = 'BEAR'
            print("🚨 [大盤體制模組] 大盤處於『空頭/高風險體制』，進場門檻提高、單筆風險減半。")
        elif latest_close > sma_20 and sma_20 > sma_60:
            regime = 'BULL'
            print("🟢 [大盤體制模組] 大盤處於『強勢多頭體制』，維持常態選股。")
        else:
            regime = 'NORMAL'
            print("🟡 [大盤體制模組] 大盤處於『震盪整理體制』，採用標準風控機制。")

        return regime, round(latest_close, 2)
    except Exception:
        return 'NORMAL', 0.0

def get_market_active_stocks():
    default_pool = {
        '2330': '台積電', '2317': '鴻海',   '2454': '聯發科', '2308': '台達電',
        '2382': '廣達',   '3231': '緯創',   '2356': '英業達', '2603': '長榮',
        '2609': '陽明',   '2615': '萬海',   '2303': '聯電',   '3037': '欣興',
        '2379': '瑞昱',   '3035': '智原',   '2408': '南亞科', '1513': '中興電',
        '1519': '華城',   '1504': '東元',   '8046': '南電',   '3661': '世芯-KY',
        '2881': '富邦金', '2882': '國泰金', '2891': '中信金', '5871': '中租-KY',
        '3017': '奇鋐',   '6669': '緯穎',   '3324': '雙鴻',   '3443': '創意'
    }
    try:
        stock_info = dl.taiwan_stock_info()
        time.sleep(0.1)
        if not stock_info.empty:
            stock_info = stock_info[stock_info['type'] == 'twse']
            stock_info = stock_info[stock_info['stock_id'].str.len() == 4]
            valid_ids = set(stock_info['stock_id'].tolist())
            filtered_pool = {k: v for k, v in default_pool.items() if k in valid_ids}
            return filtered_pool if filtered_pool else default_pool
    except Exception:
        pass
    return default_pool

# ==========================================
# 三大因子群評分（規格書第 2 節）：每個函式回傳 (分數, 細節 dict)，
# 細節 dict 供個股報告文字使用，不只是給模型內部用的中間值。
# ==========================================
def score_technical(df):
    if len(df) < 60:
        return None, None
    close = df['Close']
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    if pd.isna(ma60.iloc[-1]):
        return None, None

    c, m5, m20, m60 = close.iloc[-1], ma5.iloc[-1], ma20.iloc[-1], ma60.iloc[-1]

    if m5 > m20 > m60:
        s_trend = 100.0
        trend_label = "多頭排列 (MA5>MA20>MA60)"
    elif m5 > m20:
        s_trend = 60.0
        trend_label = "MA5>MA20，但未站上季線"
    else:
        s_trend = 0.0
        trend_label = "空頭排列 (MA5<MA20)"

    bias = (c - m20) / m20 * 100
    if bias <= 0:
        s_bias = max(0.0, 50 + bias * 10)
    elif bias <= 5:
        s_bias = 50 + bias * 10
    else:
        s_bias = max(0.0, 100 - (bias - 5) * 10)

    vol = df['Volume']
    vol_today = float(vol.iloc[-1])
    vol_ma20 = float(vol.rolling(20).mean().iloc[-1])
    vol_ratio = vol_today / (vol_ma20 + 1e-6)
    is_red = bool(close.iloc[-1] > df['Open'].iloc[-1])
    if vol_ratio >= 1.2 and is_red:
        s_vol = 100.0
    elif is_red:
        s_vol = min(100.0, vol_ratio / 1.2 * 100) * 0.6
    else:
        s_vol = min(60.0, vol_ratio / 1.2 * 60)

    ret5 = close.pct_change(5)
    window = ret5.tail(252).dropna()
    if len(window) >= 20:
        s_mom = float((window <= ret5.iloc[-1]).mean() * 100)
    else:
        s_mom = 50.0

    score = round((s_trend + s_bias + s_vol + s_mom) / 4, 1)
    details = {
        'trend_label': trend_label,
        'bias_pct': round(bias, 1),
        'vol_ratio': round(vol_ratio, 2),
        'is_red': is_red,
        'momentum_pct': round(s_mom, 1),
        'volume_today': vol_today,
        'volume_avg20': vol_ma20,
        'volume_surge_pct': round((vol_ratio - 1) * 100, 1),
    }
    return score, details

def score_fundamental(yoy, mom_turned_positive, per_series):
    if yoy is None or pd.isna(yoy):
        s_yoy = 50.0
    elif yoy > 30:
        s_yoy = 100.0
    elif yoy >= 10:
        s_yoy = 70.0
    elif yoy >= 0:
        s_yoy = 50.0
    else:
        s_yoy = max(0.0, 30 + yoy)
    if mom_turned_positive:
        s_yoy = min(100.0, s_yoy + 20)

    per_clean = per_series.dropna()
    per_window = per_clean.tail(756)  # 約 3 年交易日
    current_per = per_clean.iloc[-1] if not per_clean.empty else None
    percentile = None
    if current_per is not None and len(per_window) >= 60:
        percentile = float((per_window <= current_per).mean() * 100)
        if percentile <= 20:
            s_val = 100.0
        elif percentile >= 80:
            s_val = 0.0
        else:
            s_val = 100 - (percentile - 20) * (100 / 60)
    else:
        s_val = 50.0

    score = round(s_yoy * 0.55 + s_val * 0.45, 1)
    details = {
        'yoy': yoy,
        'mom_turned_positive': bool(mom_turned_positive),
        'per_current': float(current_per) if current_per is not None else None,
        'per_percentile': percentile,
    }
    return score, details

def score_chip(foreign_net, trust_net, volume):
    def streak_count(net_series):
        recent = net_series.tail(10)
        streak = 0
        for v in recent.iloc[::-1]:
            if v > 0:
                streak += 1
            else:
                break
        return streak

    f_streak = streak_count(foreign_net)
    t_streak = streak_count(trust_net)
    s_foreign = min(100.0, f_streak / 5 * 100)
    s_trust = min(100.0, t_streak / 5 * 100)

    combined_5d = foreign_net.tail(5).sum() + trust_net.tail(5).sum()
    volume_5d = volume.tail(5).sum()
    ratio = combined_5d / (volume_5d + 1e-6)  # 用股數比例代替金額比例（FinMind 籌碼資料為股數，非金額）
    s_strength = max(0.0, min(100.0, ratio / 0.1 * 100))

    score = round(s_foreign * 0.3 + s_trust * 0.4 + s_strength * 0.3, 1)
    details = {
        'foreign_streak': f_streak,
        'trust_streak': t_streak,
        'strength_ratio_pct': round(ratio * 100, 2),
    }
    return score, details

def build_stock_report(name, stock_id, latest_price, tech_score, tech_d,
                        fund_score, fund_d, chip_score, chip_d,
                        total_score, passed, score_threshold):
    yoy_txt = f"{fund_d['yoy']:.1f}%" if fund_d['yoy'] is not None else "無資料"
    per_txt = f"{fund_d['per_current']:.1f} 倍" if fund_d['per_current'] is not None else "無資料"
    per_pct_txt = f"（近3年百分位 {fund_d['per_percentile']:.0f}%，越低越便宜）" if fund_d['per_percentile'] is not None else ""
    mom_txt = "，月營收由負轉正" if fund_d['mom_turned_positive'] else ""

    lines = [
        f"【{name}({stock_id})】最新收盤 NT$ {latest_price}",
        f"總分 {total_score} 分（進場門檻 {score_threshold} 分）— {'✅ 達進場標準' if passed else '⚪ 尚未達標，列入觀察'}",
        f"技術面 {tech_score} 分：{tech_d['trend_label']}，乖離率 {tech_d['bias_pct']}%，"
        f"量比 {tech_d['vol_ratio']} 倍（{'收紅' if tech_d['is_red'] else '收黑'}），"
        f"近一年 5 日報酬動能百分位 {tech_d['momentum_pct']}%",
        f"基本面 {fund_score} 分：營收 YoY {yoy_txt}{mom_txt}，PE {per_txt}{per_pct_txt}",
        f"籌碼面 {chip_score} 分：外資連續買超 {chip_d['foreign_streak']} 天，"
        f"投信連續買超 {chip_d['trust_streak']} 天，"
        f"近5日法人買超力道占成交量比例 {chip_d['strength_ratio_pct']}%",
        f"成交量：今日 {tech_d['volume_today']:,.0f} 股，20日均量 {tech_d['volume_avg20']:,.0f} 股，"
        f"較均量{'放大' if tech_d['volume_surge_pct'] >= 0 else '萎縮'} {abs(tech_d['volume_surge_pct']):.1f}%",
    ]
    return "\n".join(lines)

def calculate_position_size(buy_price, sl_price, risk_pct):
    sl_pct = (buy_price - sl_price) / buy_price
    if sl_pct <= 0:
        return 0.0
    return round(min(risk_pct / sl_pct, MAX_POSITION_PCT), 1)

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 啟動 AI 多因子選股系統...")
    audit_past_predictions()

    market_regime, _ = detect_market_regime()
    if market_regime == 'BEAR':
        score_threshold = BEAR_SCORE_THRESHOLD
        risk_pct = RISK_PER_TRADE_PCT / 2
    else:
        score_threshold = ENTRY_SCORE_THRESHOLD
        risk_pct = RISK_PER_TRADE_PCT
    print(f"🎯 今日進場總分門檻: {score_threshold} 分 ｜ 單筆風險: {risk_pct}% 總資金\n")

    dynamic_stock_pool = get_market_active_stocks()
    print(f"✅ 鎖定 {len(dynamic_stock_pool)} 支熱門標的，開始執行多因子評分...\n")

    today_str = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')      # 動能百分位需要約 1 年資料
    per_start_date = (datetime.now() - timedelta(days=1150)).strftime('%Y-%m-%d')  # 估值河流位階需要約 3 年資料

    init_db()
    conn = sqlite3.connect(DB_NAME)
    results = []
    all_candidates = []

    for stock_id, name in dynamic_stock_pool.items():
        try:
            df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
            time.sleep(0.15)
            if df.empty or len(df) < 60:
                continue
            df = df.rename(columns={'max': 'High', 'min': 'Low', 'close': 'Close',
                                     'open': 'Open', 'Trading_Volume': 'Volume'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            # 籌碼面：三大法人買賣超
            df_chip = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            time.sleep(0.15)
            if not df_chip.empty:
                foreign_net_by_date = df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['buy'].sum() - \
                                      df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['sell'].sum()
                trust_net_by_date = df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['buy'].sum() - \
                                    df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['sell'].sum()
                date_key = df['date'].dt.strftime('%Y-%m-%d')
                df['Foreign_Buy'] = date_key.map(foreign_net_by_date).fillna(0)
                df['Trust_Buy'] = date_key.map(trust_net_by_date).fillna(0)
            else:
                df['Foreign_Buy'], df['Trust_Buy'] = 0.0, 0.0

            # 基本面：月營收 YoY / MoM 轉正
            latest_yoy = None
            mom_turned_positive = False
            try:
                df_rev = dl.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start_date)
                time.sleep(0.1)
                if not df_rev.empty and 'revenue' in df_rev.columns:
                    df_rev = df_rev.sort_values(['revenue_year', 'revenue_month']).reset_index(drop=True)
                    mom = df_rev['revenue'].pct_change()
                    if len(mom) >= 2:
                        mom_turned_positive = bool(mom.iloc[-2] < 0 and mom.iloc[-1] > 0)
                    if 'revenue_year_growth_ratio' in df_rev.columns and not df_rev['revenue_year_growth_ratio'].dropna().empty:
                        latest_yoy = float(df_rev['revenue_year_growth_ratio'].dropna().iloc[-1])
                        if 0 < abs(latest_yoy) < 5.0:
                            latest_yoy *= 100
            except Exception:
                pass

            # 基本面：估值河流位階（近 3 年 PE 百分位）
            per_series = pd.Series(dtype=float)
            latest_per = None
            try:
                df_per = dl.taiwan_stock_per_pbr(stock_id=stock_id, start_date=per_start_date)
                time.sleep(0.1)
                if not df_per.empty and 'PER' in df_per.columns:
                    df_per['PER'] = pd.to_numeric(df_per['PER'], errors='coerce')
                    per_series = df_per['PER']
                    per_valid = per_series.dropna()
                    latest_per = float(per_valid.iloc[-1]) if not per_valid.empty else None
            except Exception:
                pass

            # 硬性排除：營收嚴重衰退或本益比異常/過高
            if latest_yoy is not None and latest_yoy < -30.0:
                continue
            if latest_per is not None and (latest_per > 80.0 or latest_per < 0):
                continue

            tech_score, tech_detail = score_technical(df)
            if tech_score is None:
                continue
            fund_score, fund_detail = score_fundamental(latest_yoy, mom_turned_positive, per_series)
            chip_score, chip_detail = score_chip(df['Foreign_Buy'], df['Trust_Buy'], df['Volume'])

            total_score = round(tech_score * WEIGHT_TECH + fund_score * WEIGHT_FUND + chip_score * WEIGHT_CHIP, 1)
            passed = total_score >= score_threshold and min(tech_score, fund_score, chip_score) >= MIN_FACTOR_SCORE

            latest_price = round(float(df['Close'].iloc[-1]), 2)
            atr_series = get_atr14(df)
            latest_atr = atr_series.iloc[-1] if pd.notna(atr_series.iloc[-1]) else latest_price * 0.02

            buy_price = latest_price
            sl_price = round(buy_price - K_SL * latest_atr, 2)
            tp_price = round(buy_price + K_TP * latest_atr, 2)
            pos_size = calculate_position_size(buy_price, sl_price, risk_pct)

            status_label = "🔥 建議買進" if passed else "☁️ 觀望"

            if passed:
                check_df = pd.read_sql(
                    "SELECT id FROM predictions WHERE predict_date=? AND stock_id=?",
                    conn, params=(today_str, stock_id)
                )
                if check_df.empty:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO predictions
                        (predict_date, stock_id, stock_name, latest_price, buy_price, tp_price, sl_price,
                         ai_win_rate, revenue_yoy, pe_ratio, position_size, market_regime,
                         trailing_stop_price, entry_atr)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (today_str, stock_id, name, latest_price, buy_price, tp_price, sl_price,
                          total_score,
                          round(latest_yoy, 1) if latest_yoy is not None else None,
                          round(latest_per, 1) if latest_per is not None else None,
                          pos_size, market_regime, sl_price, round(float(latest_atr), 4)))

            results.append({
                '股票代碼': stock_id, '股票名稱': name, '最新收盤價': latest_price,
                '技術面': tech_score, '基本面': fund_score, '籌碼面': chip_score,
                '總分': total_score, '決策建議': status_label,
                '建議買入價': buy_price, '停利價': tp_price, '停損價': sl_price,
                '建議部位(%)': pos_size
            })

            report_text = build_stock_report(
                name, stock_id, latest_price, tech_score, tech_detail,
                fund_score, fund_detail, chip_score, chip_detail,
                total_score, passed, score_threshold
            )
            all_candidates.append({
                'stock_id': stock_id, 'name': name, 'latest_price': latest_price,
                'total_score': total_score, 'tech_score': tech_score,
                'fund_score': fund_score, 'chip_score': chip_score,
                'volume_today': tech_detail['volume_today'],
                'volume_avg20': tech_detail['volume_avg20'],
                'volume_surge_pct': tech_detail['volume_surge_pct'],
                'yoy': latest_yoy, 'per': latest_per,
                'report_text': report_text,
            })
        except Exception:
            continue

    # ===== 每日觀察報告：不論是否達進場門檻，總分前十名 + 成交量異常放大前十名，各附個別報告 =====
    cursor = conn.cursor()
    cursor.execute("DELETE FROM daily_watchlist WHERE report_date=?", (today_str,))

    def insert_watchlist(candidates, category):
        for rank, c in enumerate(candidates, start=1):
            cursor.execute('''
                INSERT INTO daily_watchlist
                (report_date, stock_id, stock_name, category, rank, latest_price,
                 total_score, tech_score, fund_score, chip_score,
                 volume, volume_avg20, volume_surge_pct, revenue_yoy, pe_ratio, report_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (today_str, c['stock_id'], c['name'], category, rank, c['latest_price'],
                  c['total_score'], c['tech_score'], c['fund_score'], c['chip_score'],
                  c['volume_today'], c['volume_avg20'], c['volume_surge_pct'],
                  round(c['yoy'], 1) if c['yoy'] is not None else None,
                  round(c['per'], 1) if c['per'] is not None else None,
                  c['report_text']))

    top_by_score = sorted(all_candidates, key=lambda c: c['total_score'], reverse=True)[:10]
    insert_watchlist(top_by_score, 'TOP_SCORE')

    top_by_volume = sorted(all_candidates, key=lambda c: c['volume_surge_pct'], reverse=True)[:10]
    insert_watchlist(top_by_volume, 'VOLUME_SURGE')

    conn.commit()
    conn.close()

    if results:
        final_df = pd.DataFrame(results).sort_values(by='總分', ascending=False).reset_index(drop=True)
        print("=== 多因子評分結果 ===")
        print(final_df.to_string(index=False))
        print(f"\n✅ 選股與部位計算完成！今日觀察名單（前十名 + 成交量異常）已記錄至 daily_watchlist。")
    else:
        print("今日無符合條件標的。")

if __name__ == "__main__":
    main()
