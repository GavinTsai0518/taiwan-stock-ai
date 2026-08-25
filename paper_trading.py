import pandas as pd
import numpy as np
import sqlite3
import warnings
from datetime import datetime, timedelta
from FinMind.data import DataLoader
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

DB_NAME = "paper_trading.db"
dl = DataLoader()

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
            pe_ratio REAL
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

        df_real = dl.taiwan_stock_daily(stock_id=stock_id, start_date=p_date)
        
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

def get_adaptive_threshold():
    conn = sqlite3.connect(DB_NAME)
    df_history = pd.read_sql("SELECT status FROM predictions WHERE status!='PENDING' ORDER BY id DESC LIMIT 20", conn)
    conn.close()

    base_threshold = 0.58

    if df_history.empty or len(df_history) < 5:
        print(f"🤖 [AI 自適應模組] 歷史平倉數據不足 (<5 筆)，維持預設進場門檻: {base_threshold*100:.1f}%")
        return base_threshold

    wins = len(df_history[df_history['status'] == 'WIN (成功停利)'])
    total = len(df_history)
    recent_win_rate = wins / total

    if recent_win_rate < 0.45:
        adjusted_threshold = 0.63
        reason = "近期實戰勝率低於 45%，提高選股標準以降低風險"
    elif recent_win_rate >= 0.65:
        adjusted_threshold = 0.56
        reason = "近期實戰勝率達 65% 以上，表現優異，適度放寬選股標準"
    else:
        adjusted_threshold = base_threshold
        reason = "近期實戰勝率維持常態"

    print(f"🤖 [AI 自適應模組] 近 20 筆真實盲測勝率: {recent_win_rate*100:.1f}% -> {reason}")
    print(f"🎯 今日 AI 動態調整後進場門檻: {adjusted_threshold*100:.1f}%\n")
    return adjusted_threshold

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
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 啟動全市場 AI 自適應+基本面量化選股系統...")
    audit_past_predictions()
    adaptive_threshold = get_adaptive_threshold()
    dynamic_stock_pool = get_market_active_stocks()

    print(f"✅ 成功鎖定 {len(dynamic_stock_pool)} 支熱門標的！開始導入基本面 (營收 YoY + PER) 與 AI 運算...\n")

    today_str = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=210)).strftime('%Y-%m-%d')
    
    init_db()
    conn = sqlite3.connect(DB_NAME)
    results = []
    feature_importance_tracker = []

    for stock_id, name in dynamic_stock_pool.items():
        try:
            # 1. 抓取日 K 線數據
            df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
            if df.empty or len(df) < 50:
                continue
            df = df.rename(columns={'max': 'High', 'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            # 2. 籌碼資料
            df_chip = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            if not df_chip.empty:
                foreign_buy = df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['buy'].sum() - \
                              df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['sell'].sum()
                trust_buy = df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['buy'].sum() - \
                            df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['sell'].sum()
                df['Foreign_Buy'] = df['date'].dt.strftime('%Y-%m-%d').map(foreign_buy).fillna(0)
                df['Trust_Buy'] = df['date'].dt.strftime('%Y-%m-%d').map(trust_buy).fillna(0)
            else:
                df['Foreign_Buy'], df['Trust_Buy'] = 0, 0

            # 3. 基本面資料：營收 (Month Revenue) 與 本益比 (PER)
            try:
                df_rev = dl.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start_date)
                if not df_rev.empty and 'revenue_year' in df_rev.columns and 'revenue_month' in df_rev.columns:
                    # 依公告日對齊：假設次月 10 號發布營收
                    df_rev['announce_date'] = pd.to_datetime(
                        df_rev['revenue_year'].astype(str) + '-' + 
                        df_rev['revenue_month'].astype(str).str.zfill(2) + '-10'
                    )
                    df_rev = df_rev.sort_values('announce_date')
                    df = pd.merge_asof(df, df_rev[['announce_date', 'revenue_year_growth_ratio']], 
                                      left_on='date', right_on='announce_date', direction='backward')
                    df['Revenue_YoY'] = df['revenue_year_growth_ratio'].fillna(0)
                else:
                    df['Revenue_YoY'] = 0
            except Exception:
                df['Revenue_YoY'] = 0

            try:
                df_per = dl.taiwan_stock_per_pbr(stock_id=stock_id, start_date=start_date)
                if not df_per.empty and 'PER' in df_per.columns:
                    df_per['date'] = pd.to_datetime(df_per['date'])
                    df = pd.merge(df, df_per[['date', 'PER']], on='date', how='left')
                    df['PER'] = df['PER'].fillna(15.0)  # 無資料給予預設值
                else:
                    df['PER'] = 15.0
            except Exception:
                df['PER'] = 15.0

            # 4. 第一層基本面硬性濾網 (Hard Filter)
            latest_yoy = df['Revenue_YoY'].iloc[-1]
            latest_per = df['PER'].iloc[-1]
            if latest_yoy < -15.0 or latest_per > 65.0 or latest_per < 0:
                continue  # 過濾營收嚴重衰退或虧損高估值股票

            # 5. 技術面與籌碼面特徵建構
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

            df['Future_3D_Return'] = df['Close'].shift(-3) / df['Close'] - 1
            df['Target'] = np.where(df['Future_3D_Return'] > 0.015, 1, 0)

            df_clean = df.dropna().copy()
            if len(df_clean) < 30:
                continue

            # 6. 擴充多因子特徵池
            features = ['MA_Diff_5_20', 'Return_1D', 'Return_5D', 'RSI_14', 'Inst_Ratio', 'Revenue_YoY', 'PER']
            X, y = df_clean[features], df_clean['Target']

            # 7. 三模型滾動重訓
            clf1 = XGBClassifier(n_estimators=40, learning_rate=0.03, max_depth=3, random_state=42)
            clf2 = LGBMClassifier(n_estimators=40, learning_rate=0.03, max_depth=3, random_state=42, verbose=-1)
            clf3 = RandomForestClassifier(n_estimators=40, max_depth=3, random_state=42)
            
            ensemble_model = VotingClassifier(estimators=[('xgb', clf1), ('lgb', clf2), ('rf', clf3)], voting='soft')
            ensemble_model.fit(X, y)

            clf1.fit(X, y)
            feature_importance_tracker.append(clf1.feature_importances_)

            up_prob = ensemble_model.predict_proba(X.tail(1))[0][1]

            latest_price = round(float(df['Close'].iloc[-1]), 2)
            support_20d = round(df['Low'].tail(20).min(), 2)
            resistance_20d = round(df['High'].tail(20).max(), 2)

            buy_price = latest_price
            tp_price = max(resistance_20d, round(buy_price * 1.035, 2))
            sl_price = min(support_20d, round(buy_price * 0.975, 2))

            status = "🔥 建議買進" if up_prob >= adaptive_threshold else "☁️ 觀望"

            if up_prob >= adaptive_threshold:
                check_df = pd.read_sql(f"SELECT * FROM predictions WHERE predict_date='{today_str}' AND stock_id='{stock_id}'", conn)
                if check_df.empty:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO predictions (predict_date, stock_id, stock_name, latest_price, buy_price, tp_price, sl_price, ai_win_rate, revenue_yoy, pe_ratio)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (today_str, stock_id, name, latest_price, buy_price, tp_price, sl_price, round(up_prob * 100, 1), round(latest_yoy, 1), round(latest_per, 1)))

            results.append({
                '股票代碼': stock_id,
                '股票名稱': name,
                '最新收盤價': latest_price,
                '營收YoY(%)': round(latest_yoy, 1),
                '本益比(PE)': round(latest_per, 1),
                'AI 勝率預估(%)': round(up_prob * 100, 1),
                '決策建議': status,
                '建議買入價': buy_price,
                '建議停利價': tp_price,
                '建議停損價': sl_price
            })
        except Exception as e:
            continue

    conn.commit()
    conn.close()

    if feature_importance_tracker:
        avg_importance = np.mean(feature_importance_tracker, axis=0)
        top_feature_idx = np.argmax(avg_importance)
        print(f"📊 [AI 因子自我診斷] 當前市場最關鍵驅動因子: 『{features[top_feature_idx]}』 (權重: {avg_importance[top_feature_idx]*100:.1f}%)\n")

    if results:
        final_df = pd.DataFrame(results)
        final_df = final_df.sort_values(by='AI 勝率預估(%)', ascending=False).reset_index(drop=True)
        print("=== 全市場 AI 自適應+基本面選股結果 ===")
        print(final_df.to_string(index=False))
        print(f"\n✅ 選股完成！勝率 >= {adaptive_threshold*100:.1f}% 之基本面優良標的已記錄至 paper_trading.db。")
    else:
        print("今日市場標的經 AI 與基本面過濾後無符合條件標的。")

if __name__ == "__main__":
    main()
