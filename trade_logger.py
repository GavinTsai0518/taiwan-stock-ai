import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from FinMind.data import DataLoader
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

DB_NAME = "paper_trading.db"
dl = DataLoader()

# ==========================================
# 1. 資料庫初始化
# ==========================================
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
            status TEXT DEFAULT 'PENDING',  -- PENDING, WIN, LOSS, EXPIRED
            real_max_price REAL DEFAULT 0,
            real_min_price REAL DEFAULT 0,
            validated_date TEXT
        )
    ''')
    conn.commit()
    conn.close()

# ==========================================
# 2. 自動結算歷史紀錄 (Forward Testing Audit)
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
        p_id, stock_id, p_date = row['id'], row['stock_id'], row['predict_date']
        tp_price, sl_price = row['tp_price'], row['sl_price']

        df_real = dl.taiwan_stock_daily(stock_id=stock_id, start_date=p_date)
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
# 3. 全市場動態抓取與自動選股
# ==========================================
def get_market_active_stocks():
    """自動從 FinMind 抓取全市場最近一個交易日成交量最大的前 50 支股票 (全動態選股)"""
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    # 抓取台股全市場日K數據
    df_all = dl.taiwan_stock_daily_info(date=datetime.now().strftime('%Y-%m-%d'))
    
    if df_all.empty:
        # 若當天盤後數據尚未完全出爐，抓最近 5 天數據取最新一天
        df_all = dl.taiwan_stock_daily(stock_id='', start_date=start_date)
    
    if df_all.empty:
        return {}

    # 取得最新的日期
    latest_date = df_all['date'].max()
    df_latest = df_all[df_all['date'] == latest_date].copy()
    
    # 轉換欄位格式與篩選流動性門檻：成交量 > 1,000 張 (1,000,000 股)
    df_latest['Trading_Volume'] = pd.to_numeric(df_latest['Trading_Volume'], errors='coerce')
    df_active = df_latest[df_latest['Trading_Volume'] > 1000000].copy()
    
    # 排除 ETF (例如 0050、0056) 與權證，只留個股 (股票代碼長度為 4 且純數字)
    df_active = df_active[df_active['stock_id'].str.len() == 4]
    df_active = df_active[df_active['stock_id'].str.isdigit()]

    # 按成交量從大到小排序，取前 40 支焦點熱門股
    df_top = df_active.sort_values(by='Trading_Volume', ascending=False).head(40)
    
    # 轉為字典 {stock_id: stock_name}
    dynamic_pool = {}
    for _, row in df_top.iterrows():
        s_id = row['stock_id']
        s_name = row.get('stock_name', s_id) # 若無名稱則用代碼代替
        dynamic_pool[s_id] = s_name
        
    return dynamic_pool

def run_daily_selection():
    init_db()
    
    print("1. 正在進行『全市場熱門焦點股』動態掃描...")
    dynamic_stock_pool = get_market_active_stocks()
    
    if not dynamic_stock_pool:
        print("未抓取到全市場動態資料，暫停今日選股。")
        return

    print(f"成功自動鎖定當日全市場資金最關注的 {len(dynamic_stock_pool)} 支熱門標的！開始進階 AI 評估...")

    today_str = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')
    conn = sqlite3.connect(DB_NAME)

    for stock_id, name in dynamic_stock_pool.items():
        try:
            df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
            if df.empty or len(df) < 50:
                continue
            df = df.rename(columns={'max': 'High', 'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'})

            # 抓取籌碼面
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

            # 三模型集成
            clf1 = XGBClassifier(n_estimators=40, learning_rate=0.03, max_depth=3, random_state=42)
            clf2 = LGBMClassifier(n_estimators=40, learning_rate=0.03, max_depth=3, random_state=42, verbose=-1)
            clf3 = RandomForestClassifier(n_estimators=40, max_depth=3, random_state=42)
            
            ensemble_model = VotingClassifier(estimators=[('xgb', clf1), ('lgb', clf2), ('rf', clf3)], voting='soft')
            ensemble_model.fit(X, y)

            up_prob = ensemble_model.predict_proba(X.tail(1))[0][1]

            # 精選看漲勝率 > 58% 的標的
            if up_prob >= 0.58:
                latest_price = round(float(df['Close'].iloc[-1]), 2)
                support_20d = round(df['Low'].tail(20).min(), 2)
                resistance_20d = round(df['High'].tail(20).max(), 2)

                buy_price = latest_price
                tp_price = max(resistance_20d, round(buy_price * 1.035, 2))
                sl_price = min(support_20d, round(buy_price * 0.975, 2))

                # 避免重複寫入
                check_df = pd.read_sql(f"SELECT * FROM predictions WHERE predict_date='{today_str}' AND stock_id='{stock_id}'", conn)
                if check_df.empty:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO predictions (predict_date, stock_id, stock_name, latest_price, buy_price, tp_price, sl_price, ai_win_rate)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (today_str, stock_id, name, latest_price, buy_price, tp_price, sl_price, round(up_prob * 100, 1)))
        except Exception:
            continue

    conn.commit()
    conn.close()

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 開始執行盤後全市場動態選股與對照紀錄...")
    audit_past_predictions()
    run_daily_selection()
    print("今日任務完成！全市場動態選股結果已記錄至 paper_trading.db。")