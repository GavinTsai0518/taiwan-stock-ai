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

# ==========================================
# 1. 資料庫初始化 (SQLite)
# ==========================================
def init_db():
    """建立 SQLite 資料庫與預測紀錄表"""
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
            status TEXT DEFAULT 'PENDING',  -- PENDING, WIN, LOSS, EXPIRED
            real_max_price REAL DEFAULT 0,
            real_min_price REAL DEFAULT 0,
            validated_date TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_date TEXT,
            historical_win_rate REAL,
            adapted_threshold REAL,
            top_feature TEXT
        )
    ''')
    conn.commit()
    conn.close()

# ==========================================
# 2. 自動結算歷史紀錄 (Audit)
# ==========================================
def audit_past_predictions():
    """自動比對過去尚未平倉的預測紀錄，檢驗是否到達停利或停損價"""
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

# ==========================================
# 3. AI 自適應門檻與自校正模組 (Adaptive Calibration)
# ==========================================
def get_adaptive_threshold():
    """依據近期盲測實戰勝率，動態校正今天的進場門檻"""
    conn = sqlite3.connect(DB_NAME)
    df_history = pd.read_sql("SELECT status FROM predictions WHERE status!='PENDING' ORDER BY id DESC LIMIT 20", conn)
    conn.close()

    base_threshold = 0.58  # 預設基準門檻 58%

    if df_history.empty or len(df_history) < 5:
        print(f"🤖 [AI 自適應模組] 歷史平倉數據不足 (<5 筆)，維持預設進場門檻: {base_threshold*100:.1f}%")
        return base_threshold

    wins = len(df_history[df_history['status'] == 'WIN (成功停利)'])
    total = len(df_history)
    recent_win_rate = wins / total

    # 自校正邏輯：根據近期勝率動態調整門檻
    if recent_win_rate < 0.45:
        adjusted_threshold = 0.63  # 勝率過低，收緊條件
        reason = "近期實戰勝率低於 45%，提高選股標準以降低風險"
    elif recent_win_rate >= 0.65:
        adjusted_threshold = 0.56  # 勝率優異，適度放寬
        reason = "近期實戰勝率達 65% 以上，表現優異，適度放寬選股標準"
    else:
        adjusted_threshold = base_threshold
        reason = "近期實戰勝率維持常態"

    print(f"🤖 [AI 自適應模組] 近 20 筆真實盲測勝率: {recent_win_rate*100:.1f}% -> {reason}")
    print(f"🎯 今日 AI 動態調整後進場門檻: {adjusted_threshold*100:.1f}%\n")
    return adjusted_threshold

# ==========================================
# 4. 獲取市場焦點標的
# ==========================================
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

# ==========================================
# 5. 主程式：滾動重訓、選股、AI 自我評估與紀錄
# ==========================================
def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 啟動全市場 AI 自適應量化選股系統...")
    
    # 1. 執行歷史對照結算
    print("1. 正在檢查與結算歷史預測紀錄 (Forward Testing Audit)...")
    audit_past_predictions()

    # 2. 獲取動態校正後的進場門檻
    print("2. 進行 AI 自適應門檻校正...")
    adaptive_threshold = get_adaptive_threshold()

    # 3. 獲取焦點個股
    print("3. 正在取得市場焦點熱門個股動態清單...")
    dynamic_stock_pool = get_market_active_stocks()

    print(f"✅ 成功鎖定 {len(dynamic_stock_pool)} 支熱門標的！開始進行『滾動式訓練 + 特徵動態評估』AI 運算...\n")

    today_str = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d') # 滾動 180 天最新資料
    
    init_db()
    conn = sqlite3.connect(DB_NAME)
    results = []
    feature_importance_tracker = []

    for stock_id, name in dynamic_stock_pool.items():
        try:
            df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
            if df.empty or len(df) < 50:
                continue
            df = df.rename(columns={'max': 'High', 'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'})

            df_chip = dl.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_date)
            if not df_chip.empty:
                foreign_buy = df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['buy'].sum() - \
                              df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['sell'].sum()
                trust_buy = df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['buy'].sum() - \
                            df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['sell'].sum()
                df['Foreign_Buy'] = df['date'].map(foreign_buy).fillna(0)
                df['Trust_Buy'] = df['date'].map(trust_buy).fillna(0)
            else:
                df['Foreign_Buy'], df['Trust_Buy'] = 0, 0

            # 特徵建構
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

            features = ['MA_Diff_5_20', 'Return_1D', 'Return_5D', 'RSI_14', 'Inst_Ratio']
            X, y = df_clean[features], df_clean['Target']

            # 三模型滾動重訓 (Rolling Re-training)
            clf1 = XGBClassifier(n_estimators=40, learning_rate=0.03, max_depth=3, random_state=42)
            clf2 = LGBMClassifier(n_estimators=40, learning_rate=0.03, max_depth=3, random_state=42, verbose=-1)
            clf3 = RandomForestClassifier(n_estimators=40, max_depth=3, random_state=42)
            
            ensemble_model = VotingClassifier(estimators=[('xgb', clf1), ('lgb', clf2), ('rf', clf3)], voting='soft')
            ensemble_model.fit(X, y)

            # 記錄各因子重要性
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

            # 僅寫入符合動態校正門檻的標的
            if up_prob >= adaptive_threshold:
                check_df = pd.read_sql(f"SELECT * FROM predictions WHERE predict_date='{today_str}' AND stock_id='{stock_id}'", conn)
                if check_df.empty:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO predictions (predict_date, stock_id, stock_name, latest_price, buy_price, tp_price, sl_price, ai_win_rate)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (today_str, stock_id, name, latest_price, buy_price, tp_price, sl_price, round(up_prob * 100, 1)))

            results.append({
                '股票代碼': stock_id,
                '股票名稱': name,
                '最新收盤價': latest_price,
                'AI 勝率預估(%)': round(up_prob * 100, 1),
                '決策建議': status,
                '建議買入價': buy_price,
                '建議停利價 (前高)': tp_price,
                '建議停損價 (支撐)': sl_price
            })
        except Exception:
            continue

    conn.commit()
    conn.close()

    # 印出因子重要性自我診斷
    if feature_importance_tracker:
        avg_importance = np.mean(feature_importance_tracker, axis=0)
        top_feature_idx = np.argmax(avg_importance)
        print(f"📊 [AI 因子自我診斷] 當前市場最關鍵驅動因子: 『{features[top_feature_idx]}』 (權重: {avg_importance[top_feature_idx]*100:.1f}%)\n")

    if results:
        final_df = pd.DataFrame(results)
        final_df = final_df.sort_values(by='AI 勝率預估(%)', ascending=False).reset_index(drop=True)
        
        print("=== 全市場 AI 自適應選股與交易價位表 ===")
        print(final_df.to_string(index=False))
        print(f"\n✅ 選股完成！勝率 >= {adaptive_threshold*100:.1f}% 之精選標的已同步記錄至 paper_trading.db。")
    else:
        print("今日市場標的經 AI 評估後無符合當前自適應門檻之標的。")

if __name__ == "__main__":
    main()