import pandas as pd
import numpy as np
import sqlite3
import warnings
import time
import os
import sys
from datetime import datetime, timedelta
from FinMind.data import DataLoader
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

DB_NAME = "paper_trading.db"

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
            market_regime TEXT DEFAULT 'NORMAL'
        )
    ''')
    conn.commit()
    conn.close()

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
        p_id, stock_id, p_date = row['id'], row['stock_id'], row['predict_date']
        tp_price, sl_price = row['tp_price'], row['sl_price']

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

        max_p = df_future['max'].max()
        min_p = df_future['min'].min()
        
        status = 'PENDING'
        if max_p >= tp_price:
            status = 'WIN (成功停利)'
        elif min_p <= sl_price:
            status = 'LOSS (觸及停損)'
        elif len(df_future) >= 5:
            status = 'EXPIRED (過期平倉)'

        if status != 'PENDING':
            cursor.execute('''
                UPDATE predictions 
                SET status=?, real_max_price=?, real_min_price=?, validated_date=?
                WHERE id=?
            ''', (status, max_p, min_p, today_str, p_id))

    conn.commit()
    conn.close()

def detect_market_regime():
    start_date = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
    try:
        df_market = dl.taiwan_stock_daily(stock_id='0050', start_date=start_date)
        time.sleep(0.1)
        if df_market.empty or len(df_market) < 30:
            return 'NORMAL', 0.0
        
        df_market = df_market.rename(columns={'close': 'Close'})
        sma_20 = df_market['Close'].rolling(20).mean().iloc[-1]
        sma_60 = df_market['Close'].rolling(60).mean().iloc[-1] if len(df_market) >= 60 else sma_20
        latest_close = df_market['Close'].iloc[-1]

        if latest_close < sma_20 and sma_20 < sma_60:
            regime = 'BEAR'
            print("🚨 [大盤體制模组] 偵測到大盤處於『空頭/高風險體制』(收盤價跌破月線/季線)！啟動全面防守降倉。")
        elif latest_close > sma_20 and sma_20 > sma_60:
            regime = 'BULL'
            print("🟢 [大盤體制模組] 大盤處於『強勢多頭體制』，系統維持常態/順勢攻擊選股。")
        else:
            regime = 'NORMAL'
            print("🟡 [大盤體制模組] 大盤處於『震盪整理體制』，採用標準風控機制。")
            
        return regime, round(latest_close, 2)
    except Exception:
        return 'NORMAL', 0.0

def get_bayesian_adaptive_threshold(market_regime):
    conn = sqlite3.connect(DB_NAME)
    df_history = pd.read_sql("SELECT status FROM predictions WHERE status!='PENDING' ORDER BY id DESC LIMIT 20", conn)
    conn.close()

    alpha_prior = 11.6
    beta_prior = 8.4
    base_threshold = 0.58

    if df_history.empty or len(df_history) < 5:
        threshold = base_threshold
    else:
        wins = len(df_history[df_history['status'] == 'WIN (成功停利)'])
        losses = len(df_history[df_history['status'] == 'LOSS (觸及停損)'])
        bayesian_win_rate = (wins + alpha_prior) / (wins + losses + alpha_prior + beta_prior)

        if bayesian_win_rate < 0.52:
            threshold = 0.62
        elif bayesian_win_rate >= 0.62:
            threshold = 0.56
        else:
            threshold = base_threshold

    if market_regime == 'BEAR':
        threshold = max(threshold, 0.64)
        print(f"🛡️ [大盤防守疊加] 受空頭體制影響，進場門檻強制提升至: {threshold*100:.1f}%\n")
    else:
        print(f"🎯 今日 AI 動態門檻: {threshold*100:.1f}%\n")

    return threshold

def calculate_kelly_position(win_rate_prob, buy_p, tp_p, sl_p):
    b = (tp_p - buy_p) / (buy_p - sl_p + 1e-6)
    p = win_rate_prob
    q = 1.0 - p

    kelly_f = (b * p - q) / (b + 1e-6)
    half_kelly = max(0.0, kelly_f / 2.0)
    capped_position = min(0.15, half_kelly)
    
    return round(capped_position * 100, 1)

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

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 啟動 AI 量化選股系統 (Token 驗證版)...")
    audit_past_predictions()
    
    market_regime, _ = detect_market_regime()
    adaptive_threshold = get_bayesian_adaptive_threshold(market_regime)
    
    dynamic_stock_pool = get_market_active_stocks()
    print(f"✅ 鎖定 {len(dynamic_stock_pool)} 支熱門標的，開始執行量化分析...\n")

    today_str = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=210)).strftime('%Y-%m-%d')
    
    init_db()
    conn = sqlite3.connect(DB_NAME)
    results = []

    for stock_id, name in dynamic_stock_pool.items():
        try:
            df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
            time.sleep(0.15)
            if df.empty or len(df) < 50:
                continue
            df = df.rename(columns={'max': 'High', 'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            # 籌碼面
            df_chip = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            time.sleep(0.15)
            if not df_chip.empty:
                foreign_buy = df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['buy'].sum() - \
                              df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['sell'].sum()
                trust_buy = df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['buy'].sum() - \
                            df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['sell'].sum()
                df['Foreign_Buy'] = df['date'].dt.strftime('%Y-%m-%d').map(foreign_buy).fillna(0)
                df['Trust_Buy'] = df['date'].dt.strftime('%Y-%m-%d').map(trust_buy).fillna(0)
            else:
                df['Foreign_Buy'], df['Trust_Buy'] = 0, 0

            # 基本面
            try:
                df_rev = dl.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start_date)
                time.sleep(0.1)
                if not df_rev.empty and 'revenue_year_growth_ratio' in df_rev.columns:
                    df_rev['announce_date'] = pd.to_datetime(
                        df_rev['revenue_year'].astype(str) + '-' + 
                        df_rev['revenue_month'].astype(str).str.zfill(2) + '-10'
                    )
                    df_rev = df_rev.sort_values('announce_date')
                    df = pd.merge_asof(df, df_rev[['announce_date', 'revenue_year_growth_ratio']], 
                                      left_on='date', right_on='announce_date', direction='backward')
                    df['Revenue_YoY'] = df['revenue_year_growth_ratio'].fillna(0)
                    if df['Revenue_YoY'].abs().max() < 5.0 and df['Revenue_YoY'].abs().max() > 0:
                        df['Revenue_YoY'] = df['Revenue_YoY'] * 100
                else:
                    df['Revenue_YoY'] = 0.0
            except Exception:
                df['Revenue_YoY'] = 0.0

            try:
                df_per = dl.taiwan_stock_per_pbr(stock_id=stock_id, start_date=start_date)
                time.sleep(0.1)
                if not df_per.empty and 'PER' in df_per.columns:
                    df_per['date'] = pd.to_datetime(df_per['date'])
                    df = pd.merge(df, df_per[['date', 'PER']], on='date', how='left')
                    df['PER'] = pd.to_numeric(df['PER'], errors='coerce').fillna(20.0)
                else:
                    df['PER'] = 20.0
            except Exception:
                df['PER'] = 20.0

            # 第一層硬性過濾
            latest_yoy = df['Revenue_YoY'].iloc[-1]
            latest_per = df['PER'].iloc[-1]
            if latest_yoy < -15.0 or latest_per > 65.0 or latest_per < 0:
                continue

            # 技術指標與 ATR
            df['SMA_5'] = df['Close'].rolling(5).mean()
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['MA_Diff_5_20'] = (df['SMA_5'] - df['SMA_20']) / df['SMA_20']
            df['Return_1D'] = df['Close'].pct_change(1)
            df['Return_5D'] = df['Close'].pct_change(5)

            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-6))))

            df['Total_Inst_Buy'] = df['Foreign_Buy'] + df['Trust_Buy']
            df['Inst_Ratio'] = df['Total_Inst_Buy'] / (df['Volume'] + 1e-6)

            tr = pd.concat([
                df['High'] - df['Low'],
                (df['High'] - df['Close'].shift(1)).abs(),
                (df['Low'] - df['Close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            df['ATR_14'] = tr.rolling(14).mean()

            # 三層屏障標籤
            target_labels = []
            for i in range(len(df)):
                if i + 3 >= len(df):
                    target_labels.append(np.nan)
                    continue
                
                entry_p = df['Close'].iloc[i]
                current_atr = df['ATR_14'].iloc[i]
                if pd.isna(current_atr) or current_atr <= 0:
                    current_atr = entry_p * 0.02

                upper_barrier = entry_p + (1.8 * current_atr)
                lower_barrier = entry_p - (1.0 * current_atr)

                future_window = df.iloc[i+1 : i+4]
                first_touch = 0

                for _, f_row in future_window.iterrows():
                    if f_row['High'] >= upper_barrier:
                        first_touch = 1
                        break
                    elif f_row['Low'] <= lower_barrier:
                        first_touch = -1
                        break
                
                target_labels.append(1 if first_touch == 1 else 0)

            df['Target'] = target_labels

            df_clean = df.dropna().copy()
            if len(df_clean) < 30:
                continue

            features = ['MA_Diff_5_20', 'Return_1D', 'Return_5D', 'RSI_14', 'Inst_Ratio', 'Revenue_YoY', 'PER']
            X, y = df_clean[features], df_clean['Target']

            if len(y.unique()) < 2:
                continue

            clf1 = XGBClassifier(n_estimators=40, learning_rate=0.03, max_depth=3, random_state=42)
            clf2 = LGBMClassifier(n_estimators=40, learning_rate=0.03, max_depth=3, random_state=42, verbose=-1)
            clf3 = RandomForestClassifier(n_estimators=40, max_depth=3, random_state=42)
            
            ensemble_model = VotingClassifier(estimators=[('xgb', clf1), ('lgb', clf2), ('rf', clf3)], voting='soft')
            ensemble_model.fit(X, y)

            up_prob = ensemble_model.predict_proba(X.tail(1))[0][1]

            latest_price = round(float(df['Close'].iloc[-1]), 2)
            latest_atr = df['ATR_14'].iloc[-1] if not pd.isna(df['ATR_14'].iloc[-1]) else latest_price * 0.02

            buy_price = latest_price
            tp_price = round(buy_price + (1.8 * latest_atr), 2)
            sl_price = round(buy_price - (1.0 * latest_atr), 2)

            pos_size = calculate_kelly_position(up_prob, buy_price, tp_price, sl_price)

            status = "🔥 建議買進" if up_prob >= adaptive_threshold else "☁️ 觀望"

            if up_prob >= adaptive_threshold:
                check_df = pd.read_sql(f"SELECT * FROM predictions WHERE predict_date='{today_str}' AND stock_id='{stock_id}'", conn)
                if check_df.empty:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO predictions (predict_date, stock_id, stock_name, latest_price, buy_price, tp_price, sl_price, ai_win_rate, revenue_yoy, pe_ratio, position_size, market_regime)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (today_str, stock_id, name, latest_price, buy_price, tp_price, sl_price, round(up_prob * 100, 1), round(latest_yoy, 1), round(latest_per, 1), pos_size, market_regime))

            results.append({
                '股票代碼': stock_id,
                '股票名稱': name,
                '最新收盤價': latest_price,
                'AI 勝率(%)': round(up_prob * 100, 1),
                '建議部位(%)': pos_size,
                '決策建議': status,
                '建議買入價': buy_price,
                '建議停利價 (ATR)': tp_price,
                '建議停損價 (ATR)': sl_price
            })
        except Exception:
            continue

    conn.commit()
    conn.close()

    if results:
        final_df = pd.DataFrame(results)
        final_df = final_df.sort_values(by='AI 勝率(%)', ascending=False).reset_index(drop=True)
        print("=== 系統運算結果 (Token 授權版) ===")
        print(final_df.to_string(index=False))
        print(f"\n✅ 選股與部位計算完成！結果已記錄至 paper_trading.db。")
    else:
        print("今日無符合條件標的。")

if __name__ == "__main__":
    main()
