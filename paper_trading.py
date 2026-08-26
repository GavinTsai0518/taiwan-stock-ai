import pandas as pd
import numpy as np
import sqlite3
import warnings
import time
import os
import sys
import io
import requests
import yfinance as yf
from datetime import datetime, timedelta
from FinMind.data import DataLoader

warnings.filterwarnings('ignore')

DB_NAME = "paper_trading.db"

# ===== 可調參數（依「AI 選股引擎規格書」第 9 節建議預設值，之後可依實戰結果調整）=====
# 技術/基本/籌碼/跨市場四因子權重，加總為 1.0（新增跨市場因子後，其餘三個因子權重等比例下修）
WEIGHT_TECH = 0.35
WEIGHT_FUND = 0.25
WEIGHT_CHIP = 0.25
WEIGHT_MACRO = 0.15
ENTRY_SCORE_THRESHOLD = 70      # 一般/多頭體制進場總分門檻
BEAR_SCORE_THRESHOLD = 80       # 空頭體制進場總分門檻（提高）

# ===== 自我訓練功能參數（見 compute_self_training_metrics）=====
SELF_TRAIN_MIN_SAMPLES = 20        # 累積驗證筆數（WIN+LOSS）少於此數，樣本太少不足以調整，維持固定門檻
SELF_TRAIN_MIN_BUCKET_SAMPLES = 10  # 每個分數級距至少要有這麼多筆才拿來判斷勝率，避免小樣本雜訊誤導
SELF_TRAIN_TARGET_WIN_RATE = 0.55   # 目標勝率：找出「總分 >= 這個門檻」的歷史勝率能達到多少的最低門檻
SELF_TRAIN_THRESHOLD_STEPS = (0, 5, 10, 15)  # 在固定門檻基礎上往上測試的級距（只會往上調，不會往下調）
MIN_FACTOR_SCORE = 50           # 三因子單項一票否決門檻（P0 優化：40→50，原本 40 分只是平均水準，太容易放行弱因子）
MIN_DAILY_TURNOVER = 100_000_000  # 流動性門檻：日成交金額 < 1 億視為易滑點，直接排除
DEFAULT_BETA = 1.0               # 個股 Beta，目前無現成資料來源，先用市場平均值 1.0（等於停用 Beta 相關加嚴條件）
K_SL = 1.75                     # 停損 ATR 倍數（規格建議 1.5~2.0，取中間值）
K_TP = 3.0                      # 停利 ATR 倍數
RISK_PER_TRADE_PCT = 1.0        # 單筆風險占總資金 %（空頭體制會減半）
MAX_POSITION_PCT = 10.0         # 單檔部位上限 %
MARKET_PROXY_ID = '0050'        # FinMind 的 taiwan_stock_daily 沒有加權指數代碼，改用追蹤大盤的 0050 ETF 代理
CANDIDATE_POOL_SIZE = 150       # 候選股數：依 TWSE 全市場成交量排序取前 N 名進入深度多因子評分
TWSE_STOCK_DAY_ALL_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"  # 全市場單日快照，免費無需權杖
TWSE_T86_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"  # 三大法人買賣超日報，支援歷史日期回溯查詢
MOPS_REVENUE_URL_TMPL = "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_{roc_year}_{month}_0.html"  # 公開資訊觀測站上市公司月營收
TWSE_REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

class TwseRateLimitError(Exception):
    """T86 短時間內請求太多次會被 TWSE 的反爬蟲機制擋下，回傳「FOR SECURITY REASONS」的阻擋頁而不是
    JSON 資料。這跟「當天是假日、本來就沒資料」是完全不同的情況，不該被靜默吞掉當成同一種結果——
    否則像 ml_trend_model.py 的歷史回溯會在被擋下之後，後面幾百次請求全部拿到空結果卻毫無警示，
    看起來只是「那些天剛好沒交易」，實際上是資料被整批打壞。呼叫端應該看到這個例外就提早停止重試。"""
    pass

# FinMind Token 一律從環境變數讀取（GitHub Actions 用 repo secret 注入），不再寫死於程式碼中。
# DataLoader() 本身不用權杖也不連網，可以放在模組層級；真正的登入動作包成函式、只在
# 直接執行本檔案時呼叫，這樣其他腳本（例如 cross_market_validation.py）才能安全地
# import 這裡的評分函式重複使用，不會被迫觸發 FinMind 登入或要求設定 FINMIND_TOKEN。
dl = DataLoader()

def ensure_finmind_login():
    token = os.environ.get("FINMIND_TOKEN")
    if not token:
        print("❌ 未設定 FINMIND_TOKEN 環境變數，無法登入 FinMind，中止執行。")
        sys.exit(1)
    try:
        dl.login_by_token(api_token=token)
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
                     "ALTER TABLE predictions ADD COLUMN entry_atr REAL",
                     # 自我訓練功能用：存下當初進場時各子維度分數，之後才能回頭分析
                     # 「哪個子維度的分數跟實際勝負最相關」（見 compute_self_training_metrics）。
                     "ALTER TABLE predictions ADD COLUMN tech_score REAL",
                     "ALTER TABLE predictions ADD COLUMN fund_score REAL",
                     "ALTER TABLE predictions ADD COLUMN chip_score REAL",
                     "ALTER TABLE predictions ADD COLUMN macro_score REAL"):
        try:
            cursor.execute(col_sql)
        except sqlite3.OperationalError:
            pass

    # 自我訓練功能：記錄每次評估時的歷史勝率、依勝率動態調整出的進場門檻、以及目前
    # 最能區分勝負的子維度分數。見 compute_self_training_metrics()。
    # 這張表其實已經存在於正式資料庫（早期設計就有這張表，但從沒有程式碼真的寫過/讀過），
    # 只有舊的 5 欄，跟 predictions 表當初遇到的問題一模一樣：CREATE TABLE IF NOT EXISTS
    # 對已存在的表是 no-op，所以一樣要用 ALTER TABLE 補欄位，否則 INSERT 會因為缺
    # resolved_count 欄位而失敗。
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_date TEXT,
            resolved_count INTEGER,
            historical_win_rate REAL,
            adapted_threshold REAL,
            top_feature TEXT
        )
    ''')
    try:
        cursor.execute("ALTER TABLE model_metrics ADD COLUMN resolved_count INTEGER")
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

# ==========================================
# 自我訓練功能：用已驗證的歷史勝負，動態調整今天的進場門檻
# ==========================================
def compute_self_training_metrics():
    """讀取 predictions 表裡已經驗證完（WIN/LOSS，不含 PENDING/EXPIRED）的歷史紀錄，
    算出目前的實際勝率，並嘗試找出一個「調整後門檻」：在固定門檻基礎上，依序測試
    +0/+5/+10/+15 分的級距，找出「總分 >= 該級距」歷史勝率能達到 SELF_TRAIN_TARGET_WIN_RATE
    的最低級距。這個調整只會讓門檻變嚴（往上調），不會自動放寬——就算歷史勝率看起來很漂亮，
    也不代表未來會一樣，用小樣本的漂亮數字去放寬風控門檻，過擬合/自我強化錯覺的風險比潛在
    好處大，所以刻意設計成只能收緊，不能放寬。

    同時比較 WIN 組跟 LOSS 組在 tech/fund/chip/macro 四個子分數的平均差距，抓出目前最能
    區分勝負的子維度存成 top_feature，純粹是診斷資訊，不會拿來自動調整權重（調整權重公式
    本身風險更高，需要更多資料和更謹慎的驗證，目前只做「調整進場門檻」這一步）。

    回傳 (adapted_threshold, historical_win_rate)：
    - 樣本數 < SELF_TRAIN_MIN_SAMPLES 時，adapted_threshold 回傳 None（代表沿用固定門檻，
      不做任何調整，因為樣本太少時任何調整都只是雜訊，不是真訊號）。
    - historical_win_rate 樣本不足時也回傳 None，純粹供 log 顯示用。
    """
    init_db()
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql(
        "SELECT * FROM predictions WHERE status LIKE 'WIN%' OR status LIKE 'LOSS%'", conn
    )

    resolved_count = len(df)
    today_str = datetime.now().strftime('%Y-%m-%d')

    if resolved_count < SELF_TRAIN_MIN_SAMPLES:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO model_metrics (eval_date, resolved_count, historical_win_rate, adapted_threshold, top_feature)
            VALUES (?, ?, ?, ?, ?)
        ''', (today_str, resolved_count, None, None,
              f"樣本不足（{resolved_count}/{SELF_TRAIN_MIN_SAMPLES}），尚未開始自我調整"))
        conn.commit()
        conn.close()
        return None, None

    df['is_win'] = df['status'].str.startswith('WIN')
    historical_win_rate = float(df['is_win'].mean())

    adapted_threshold = None
    for step in SELF_TRAIN_THRESHOLD_STEPS:
        level = ENTRY_SCORE_THRESHOLD + step
        bucket = df[df['ai_win_rate'] >= level]
        if len(bucket) < SELF_TRAIN_MIN_BUCKET_SAMPLES:
            continue
        bucket_win_rate = float(bucket['is_win'].mean())
        if bucket_win_rate >= SELF_TRAIN_TARGET_WIN_RATE:
            adapted_threshold = level
            break

    top_feature = None
    sub_cols = ['tech_score', 'fund_score', 'chip_score', 'macro_score']
    df_sub = df.dropna(subset=sub_cols)
    if len(df_sub) >= SELF_TRAIN_MIN_BUCKET_SAMPLES:
        win_means = df_sub[df_sub['is_win']][sub_cols].mean()
        loss_means = df_sub[~df_sub['is_win']][sub_cols].mean()
        gap = (win_means - loss_means).sort_values(ascending=False)
        if not gap.empty:
            top_feature = f"{gap.index[0]}（勝負組平均差 {gap.iloc[0]:+.1f} 分）"

    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO model_metrics (eval_date, resolved_count, historical_win_rate, adapted_threshold, top_feature)
        VALUES (?, ?, ?, ?, ?)
    ''', (today_str, resolved_count, historical_win_rate, adapted_threshold, top_feature))
    conn.commit()
    conn.close()
    return adapted_threshold, historical_win_rate

def detect_market_regime():
    """回傳 (趨勢體制, 波動率體制, 大盤收盤價)。
    趨勢體制 BULL/BEAR/NORMAL：決定進場總分門檻與單筆風險。
    波動率體制 LOW/NORMAL/HIGH（P0 優化新增）：決定技術面子權重與停損停利 ATR 倍數。"""
    start_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')  # 波動率百分位需要較長歷史
    try:
        df_market = dl.taiwan_stock_daily(stock_id=MARKET_PROXY_ID, start_date=start_date)
        time.sleep(0.1)
        if df_market.empty or len(df_market) < 30:
            print(f"⚠️ [大盤體制模組] 大盤代理股 {MARKET_PROXY_ID} 資料為空或不足 30 筆（可能是 FinMind 額度用盡），退回 NORMAL 體制。")
            return 'NORMAL', 'NORMAL', 0.0

        df_market = df_market.rename(columns={'close': 'Close'})
        df_market['date'] = pd.to_datetime(df_market['date'])
        df_market = df_market.sort_values('date').reset_index(drop=True)

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

        # 波動率體制：20日已實現波動度，相對自身近半年分布的百分位
        ret = df_market['Close'].pct_change()
        realized_vol = ret.rolling(20).std()
        vol_window = realized_vol.tail(120).dropna()
        volatility_regime = 'NORMAL'
        if len(vol_window) >= 30 and pd.notna(realized_vol.iloc[-1]):
            vol_percentile = float((vol_window <= realized_vol.iloc[-1]).mean() * 100)
            if vol_percentile >= 67:
                volatility_regime = 'HIGH'
            elif vol_percentile <= 33:
                volatility_regime = 'LOW'
            print(f"📈 [波動率模組] 大盤 20 日波動度處於近半年 {vol_percentile:.0f} 百分位 → {volatility_regime} 體制。")

        return regime, volatility_regime, round(latest_close, 2)
    except Exception:
        return 'NORMAL', 'NORMAL', 0.0

def _curated_stock_universe():
    # 免費 FinMind 帳號（register 等級）不能用 stock_id='' 一次查全市場（要 Sponsor 付費等級才開放，
    # 實測會直接噴 "Your level is register" 錯誤），所以改成手動列一份涵蓋各產業的候選清單，
    # 執行時再用 taiwan_stock_info() 驗證代碼是否還存在、順便拿正確的股名，錯誤/下市的代碼會被自動濾掉。
    return {
        # 半導體 / 電子代工 / PC 生態系
        '2330': '台積電', '2317': '鴻海',   '2454': '聯發科', '2308': '台達電',
        '2382': '廣達',   '3231': '緯創',   '2356': '英業達', '2303': '聯電',
        '3037': '欣興',   '2379': '瑞昱',   '3035': '智原',   '2408': '南亞科',
        '8046': '南電',   '3661': '世芯-KY', '3017': '奇鋐',   '6669': '緯穎',
        '3324': '雙鴻',   '3443': '創意',   '2357': '華碩',   '2353': '宏碁',
        '2377': '微星',   '2376': '技嘉',   '3706': '神達',   '6488': '環球晶',
        '3711': '日月光投控', '2449': '京元電子', '2451': '創見', '5347': '世界先進',
        '3105': '穩懋',   '8299': '群聯',   '3653': '健策',   '8069': '元太',
        '3529': '力旺',   '5269': '祥碩',   '3034': '聯詠',   '8016': '矽創',
        '3532': '台勝科', '4919': '新唐',   '3227': '原相',

        # 金融保險
        '2881': '富邦金', '2882': '國泰金', '2891': '中信金', '5871': '中租-KY',
        '2884': '玉山金', '2885': '元大金', '2886': '兆豐金', '2887': '台新金',
        '2892': '第一金', '2880': '華南金', '2883': '開發金', '2890': '永豐金',
        '5880': '合庫金', '2801': '彰銀',   '2809': '京城銀', '2812': '台中銀',
        '2834': '臺企銀', '2836': '高雄銀', '2838': '聯邦銀', '2845': '遠東銀',
        '2849': '安泰銀', '2867': '三商壽', '5876': '上海商銀',

        # 航運
        '2603': '長榮',   '2609': '陽明',   '2615': '萬海',   '2606': '裕民',
        '2610': '華航',   '2618': '長榮航', '5608': '四維航', '2637': '慧洋-KY',

        # 鋼鐵 / 塑化 / 水泥
        '2002': '中鋼',   '1101': '台泥',   '1102': '亞泥',   '1301': '台塑',
        '1303': '南亞',   '1326': '台化',   '6505': '台塑化', '2027': '大成鋼',
        '2014': '中鴻',

        # 電機 / 重電
        '1513': '中興電', '1519': '華城',   '1504': '東元',   '1503': '士電',

        # 電信 / 網通
        '2412': '中華電', '3045': '台灣大', '4904': '遠傳',   '2345': '智邦',
        '5388': '中磊',   '4906': '正文',

        # 面板 / PCB / IC 通路
        '3481': '群創',   '2409': '友達',   '2313': '華通',   '6213': '聯茂',

        # 零售 / 消費
        '2912': '統一超', '1216': '統一',   '2903': '遠百',   '9910': '豐泰',
        '2731': '六福',   '8422': '可寧衛', '2915': '潤泰全', '8454': '富邦媒',

        # 營建 / 資產
        '2542': '興富發', '2547': '日勝生', '2506': '太設',   '5522': '遠雄',
        '2520': '冠德',

        # 生技醫療
        '4137': '麗豐-KY', '4174': '浩鼎',  '6446': '藥華藥', '1795': '美時',

        # 汽車
        '2201': '裕隆',   '2207': '和泰車', '1319': '東陽',

        # 化工
        '1710': '東聯',   '1717': '長興',   '4720': '亞聚',

        # 紡織 / 民生消費
        '1402': '遠東新', '1440': '南紡',   '1476': '儒鴻',   '9904': '寶成',
        '2105': '正新',

        # 食品
        '1210': '大成',   '1227': '佳格',   '1229': '聯華',

        # 造紙 / 電線電纜
        '1904': '正隆',   '1907': '永豐餘', '1609': '大亞',

        # 遊戲 / 光學 / 設備
        '3293': '鈊象',   '3406': '玉晶光', '6231': '系微',   '6244': '茂迪',
        '3576': '聯合再生', '3006': '晶豪科', '5471': '松翰',

        # 觀光 / 特用材料 / 半導體設備
        '2707': '晶華',   '2059': '川湖',   '4551': '智伸科', '1560': '中砂',
        '6182': '合晶',   '3583': '辛耘',   '8121': '越峰',   '6274': '台燿',
        '6414': '樺漢',
    }

def fetch_twse_market_snapshot():
    """全市場當日快照（TWSE OpenAPI STOCK_DAY_ALL）：免費、不需權杖、單次請求取得約 1,300+ 檔
    上市證券的當日 OHLC 與成交量。只回傳「最新一個交易日」，不支援歷史日期查詢。
    用 4 碼純數字且不以 00 開頭排除 ETF / 權證（實測 00 開頭的 4 碼代碼只有 0050/0056 等 8 檔舊制 ETF，
    新制 ETF 代碼多為 5~6 碼，天然就會被 4 碼過濾掉）。"""
    resp = requests.get(TWSE_STOCK_DAY_ALL_URL, headers={"Accept": "application/json"}, timeout=20)
    resp.raise_for_status()
    rows = resp.json()
    result = []
    for r in rows:
        code = str(r.get('Code', ''))
        if len(code) == 4 and code.isdigit() and not code.startswith('00'):
            try:
                volume = int(r['TradeVolume'])
            except (ValueError, TypeError, KeyError):
                continue
            result.append({'stock_id': code, 'stock_name': str(r.get('Name', code)).strip(), 'volume': volume})
    return result

def get_market_active_stocks():
    """候選股清單：改用 TWSE OpenAPI 全市場快照，依當日成交量排序取前 CANDIDATE_POOL_SIZE 名。
    免費、無流量限制、不佔用 FinMind 額度，真正反映全市場當下最活躍的股票（不是固定清單）。
    抓取失敗時退回手動維護的產業分散候選清單。"""
    try:
        snapshot = fetch_twse_market_snapshot()
        if not snapshot:
            print("⚠️ TWSE 全市場快照為空，退回備援清單。")
            return _curated_stock_universe()
        snapshot.sort(key=lambda r: r['volume'], reverse=True)
        top = snapshot[:CANDIDATE_POOL_SIZE]
        return {r['stock_id']: r['stock_name'] for r in top}
    except Exception as e:
        print(f"⚠️ TWSE 全市場快照抓取失敗（{e}），退回備援清單。")
        return _curated_stock_universe()

def fetch_twse_institutional_day(date_str):
    """單日全市場三大法人買賣超（TWSE T86），date_str 格式 YYYYMMDD。
    回傳 {stock_id: {'foreign': 外資淨買超股數, 'trust': 投信淨買超股數, 'dealer': 自營商淨買超股數}}；
    非交易日（假日）一律回傳空 dict，由呼叫端自行跳過。若被 TWSE 反爬蟲機制擋下（短時間內請求太多次），
    拋出 TwseRateLimitError 讓呼叫端知道要停止重試，而不是把「被擋」誤判成「這天沒資料」。"""
    resp = requests.get(
        TWSE_T86_URL, params={'date': date_str, 'selectType': 'ALL', 'response': 'json'},
        headers=TWSE_REQUEST_HEADERS, timeout=20
    )
    if resp.status_code in (307, 403, 428) or 'SECURITY REASONS' in resp.text:
        raise TwseRateLimitError(f"TWSE T86 請求被擋下（status={resp.status_code}），可能觸發了流量限制。")
    try:
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}
    if data.get('stat') != 'OK':
        return {}
    result = {}
    for row in data.get('data', []):
        try:
            code = str(row[0]).strip()
            foreign_net = int(str(row[4]).replace(',', ''))
            trust_net = int(str(row[10]).replace(',', ''))
            dealer_net = int(str(row[11]).replace(',', ''))
        except (IndexError, ValueError, AttributeError):
            continue
        result[code] = {'foreign': foreign_net, 'trust': trust_net, 'dealer': dealer_net}
    return result

def fetch_recent_institutional_data(num_trading_days=12, max_lookback_days=25):
    """從今天往回抓最近 num_trading_days 個交易日的全市場三大法人資料。
    關鍵優化：一天只打 1 次 API（全市場一起拿），不是一支股票打 1 次——
    對 150 支候選股來說，原本要 150 次 FinMind 呼叫，現在只要 ~12 次 TWSE 呼叫。
    回傳 {YYYYMMDD: {stock_id: {...}}}，非交易日自動跳過。"""
    results = {}
    d = datetime.now()
    tries = 0
    while len(results) < num_trading_days and tries < max_lookback_days:
        date_str = d.strftime('%Y%m%d')
        try:
            day_data = fetch_twse_institutional_day(date_str)
        except TwseRateLimitError as e:
            print(f"⚠️ {e} 停止繼續回溯，改用目前已抓到的 {len(results)} 天。")
            break
        if day_data:
            results[date_str] = day_data
        d -= timedelta(days=1)
        tries += 1
        time.sleep(0.3)
    return results

def build_institutional_series(inst_data_by_date, stock_id):
    """把 {日期: {股票代碼: {...}}} 轉成單一股票的 (外資, 投信, 自營商) 三個 pandas Series，
    依日期由舊到新排序（符合 score_chip() 的 tail() 語意：最後一筆是最新一天）。
    缺資料的日期一律補 0（等同「當天無明顯買賣超」，不影響 streak 計算的保守性）。"""
    dates_sorted = sorted(inst_data_by_date.keys())
    foreign_vals, trust_vals, dealer_vals = [], [], []
    for d in dates_sorted:
        rec = inst_data_by_date[d].get(stock_id)
        foreign_vals.append(rec['foreign'] if rec else 0)
        trust_vals.append(rec['trust'] if rec else 0)
        dealer_vals.append(rec['dealer'] if rec else 0)
    return pd.Series(foreign_vals), pd.Series(trust_vals), pd.Series(dealer_vals)

def _fetch_mops_revenue_month(roc_year, month):
    """抓 MOPS 公開資訊觀測站某一個月「全部上市公司」的月營收（一次請求涵蓋整個市場，網頁為 Big5 編碼）。
    回傳 {stock_id: {'revenue': 當月營收, 'yoy_pct': 去年同月增減%, 'mom_pct': 上月比較增減%}}，
    該月尚未公告或格式解析失敗一律回傳空 dict。"""
    url = MOPS_REVENUE_URL_TMPL.format(roc_year=roc_year, month=month)
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        resp.raise_for_status()
        html = resp.content.decode('big5hkscs', errors='replace')
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        return {}
    result = {}
    for t in tables:
        if t.shape[1] != 11:
            continue
        for _, row in t.iterrows():
            try:
                code = str(row.iloc[0]).strip()
                if not code or not code[0].isdigit():
                    continue
                revenue = float(str(row.iloc[2]).replace(',', ''))
                mom_pct = float(str(row.iloc[5]).replace(',', ''))
                yoy_pct = float(str(row.iloc[6]).replace(',', ''))
            except (ValueError, IndexError):
                continue
            result[code] = {'revenue': revenue, 'yoy_pct': yoy_pct, 'mom_pct': mom_pct}
    return result

def fetch_mops_revenue_snapshot():
    """全市場最新一期月營收（MOPS），取代原本逐股呼叫 FinMind taiwan_stock_month_revenue。
    只需要 2 次請求（當期 + 上一期，用來判斷 MoM 是否由負轉正），不是 1 次/股。
    月營收依規定次月 10 日前公告，從「上個月」開始找，找不到（尚未公告/假期）就再往前一個月，最多試 3 次。
    回傳 {stock_id: {'yoy_pct': 去年同月增減%, 'mom_turned_positive': bool}}。"""
    d = datetime.now().replace(day=1) - timedelta(days=1)  # 上個月最後一天，避免抓到「本月」這種一定還沒公告的期別
    y, m = d.year, d.month
    current_data = {}
    for _ in range(3):
        current_data = _fetch_mops_revenue_month(y - 1911, m)
        if current_data:
            break
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
        time.sleep(0.3)

    if not current_data:
        print("⚠️ MOPS 月營收抓取失敗（連續 3 個月都找不到資料），基本面營收因子將以無資料處理。")
        return {}

    py, pm = (y, m - 1) if m > 1 else (y - 1, 12)
    prior_data = _fetch_mops_revenue_month(py - 1911, pm)

    result = {}
    for code, rec in current_data.items():
        prior_mom = prior_data.get(code, {}).get('mom_pct')
        mom_turned_positive = prior_mom is not None and prior_mom < 0 and rec['mom_pct'] > 0
        result[code] = {'yoy_pct': rec['yoy_pct'], 'mom_turned_positive': mom_turned_positive}
    return result

def fetch_cross_market_signal():
    """跨市場領先指標：抓美股（S&P500 + 那斯達克平均）與日股（日經225）「最近一個已完整收盤」交易日的
    報酬率。這支排程在台股收盤後（15:15）執行時，美股通常還沒收盤，抓到的會是「前一個」完整交易日；
    日股當天多半已收盤，抓到的是「當天」——兩者合起來反映亞股/全球風險偏好對台股的外部影響，
    是全市場共用的單一訊號，一次抓取即可，不需要每支股票各抓一次。
    抓取失敗回傳 (None, None)，由 score_macro() 自行處理成中性分數，不影響其他因子。"""
    try:
        data = yf.download(['^GSPC', '^IXIC', '^N225'], period='10d', progress=False)['Close']
    except Exception as e:
        print(f"⚠️ 跨市場指數抓取失敗（{e}），跨市場因子將以中性值處理。")
        return None, None

    def _latest_return(col):
        if col not in data.columns:
            return None
        s = data[col].dropna()
        if len(s) < 2:
            return None
        return float((s.iloc[-1] / s.iloc[-2] - 1) * 100)

    us_sp500 = _latest_return('^GSPC')
    us_nasdaq = _latest_return('^IXIC')
    jp_nikkei = _latest_return('^N225')

    us_vals = [v for v in (us_sp500, us_nasdaq) if v is not None]
    us_return = sum(us_vals) / len(us_vals) if us_vals else None

    return us_return, jp_nikkei

# ==========================================
# 三大因子群評分（P0 優化版）：每個函式回傳 (分數, 細節 dict)，
# 細節 dict 供個股報告文字使用，不只是給模型內部用的中間值。
# ==========================================
def score_macro(us_return_pct, jp_return_pct):
    """跨市場領先指標評分：美股、日股報酬率各自換算 0~100 分（±2.5% 觸頂/觸底，中間線性），
    美股權重較高（0.6）因為對台股電子權值股影響通常更直接，日股權重 0.4。
    任一邊抓不到資料時該邊給中性 50 分，不會讓整個因子直接失效。"""
    def _sub_score(pct):
        if pct is None:
            return 50.0
        return max(0.0, min(100.0, 50 + pct * 20))

    us_score = _sub_score(us_return_pct)
    jp_score = _sub_score(jp_return_pct)
    score = round(us_score * 0.6 + jp_score * 0.4, 1)
    details = {'us_return_pct': us_return_pct, 'jp_return_pct': jp_return_pct}
    return score, details
def score_technical(df, volatility_regime='NORMAL'):
    """P0 優化：子維度依大盤波動率動態加權、乖離率改分段曲線、動能融合短期(60天)+長期(252天)。"""
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

    # 乖離率：分段曲線，±5% 內線性到頂，超過後用次方緩衝減分（避免極端值評分失真）
    bias = (c - m20) / m20 * 100
    if abs(bias) <= 5:
        s_bias = 50 + abs(bias) * 10
    elif bias > 5:
        s_bias = max(0.0, 100 - (bias - 5) ** 1.5 * 3)
    else:
        s_bias = max(0.0, 100 - (abs(bias) - 5) ** 1.5 * 3)

    vol = df['Volume']
    vol_today = float(vol.iloc[-1])
    vol_ma20 = float(vol.rolling(20).mean().iloc[-1])
    vol_ratio = vol_today / (vol_ma20 + 1e-6)
    is_red = bool(close.iloc[-1] > df['Open'].iloc[-1])
    if vol_ratio >= 1.2 and is_red:
        s_vol_base = 100.0
    elif is_red:
        s_vol_base = min(100.0, vol_ratio / 1.2 * 100) * 0.6
    else:
        s_vol_base = min(60.0, vol_ratio / 1.2 * 60)
    vol_multiplier = 0.9 if volatility_regime == 'HIGH' else (1.1 if volatility_regime == 'LOW' else 1.0)
    s_vol = min(100.0, s_vol_base * vol_multiplier)

    # 動能：短期(60天,60%權重)+長期(252天,40%權重)百分位融合，短期更敏銳、長期防雜訊
    ret5 = close.pct_change(5)
    window_short = ret5.tail(60).dropna()
    s_mom_short = float((window_short <= ret5.iloc[-1]).mean() * 100) if len(window_short) >= 20 else 50.0
    window_long = ret5.tail(252).dropna()
    s_mom_long = float((window_long <= ret5.iloc[-1]).mean() * 100) if len(window_long) >= 20 else 50.0
    s_mom = s_mom_short * 0.6 + s_mom_long * 0.4

    # 子維度權重：波動率高時加重趨勢（防守）、波動率低時加重量能與動能（順勢）
    if volatility_regime == 'HIGH':
        w_trend, w_bias, w_vol, w_mom = 0.40, 0.25, 0.20, 0.15
    elif volatility_regime == 'LOW':
        w_trend, w_bias, w_vol, w_mom = 0.30, 0.20, 0.25, 0.25
    else:
        w_trend, w_bias, w_vol, w_mom = 0.35, 0.25, 0.20, 0.20

    score = round(s_trend * w_trend + s_bias * w_bias + s_vol * w_vol + s_mom * w_mom, 1)
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

def score_fundamental(yoy, mom_turned_positive, per_series,
                       gross_margin=None, debt_ratio=None, eps_growth_expected=None):
    """P0 優化：YoY 曲線更陡峭區分增速優劣、PE 窗口縮短至 2 年更敏銳、
    加入 PEG 比率與毛利率/負債比品質篩選（目前 FinMind 免費版無這些資料，缺值時自動跳過不計分）。"""
    if yoy is None or pd.isna(yoy):
        s_yoy = 50.0
    elif yoy > 50:
        s_yoy = 100.0
    elif yoy > 30:
        s_yoy = 85.0
    elif yoy > 15:
        s_yoy = 70.0
    elif yoy >= 5:
        s_yoy = 55.0
    elif yoy >= 0:
        s_yoy = 40.0
    else:
        s_yoy = max(0.0, 20 + yoy)
    if mom_turned_positive:
        s_yoy = min(100.0, s_yoy + 20)

    margin_penalty = 1.0
    if gross_margin is not None:
        if gross_margin < 10:
            margin_penalty = 0.6
        elif gross_margin < 15:
            margin_penalty = 0.8

    debt_penalty = 1.0
    if debt_ratio is not None:
        if debt_ratio > 70:
            debt_penalty = 0.5
        elif debt_ratio > 50:
            debt_penalty = 0.8

    per_clean = per_series.dropna()
    per_window = per_clean.tail(504)  # 2 年，比原本 3 年窗口更敏銳
    current_per = per_clean.iloc[-1] if not per_clean.empty else None
    percentile = None

    peg_score = 50.0
    if eps_growth_expected is not None and eps_growth_expected > 0 and current_per is not None:
        peg = float(current_per) / eps_growth_expected
        if peg < 1.0:
            peg_score = 100.0
        elif peg < 1.5:
            peg_score = 75.0
        elif peg < 2.0:
            peg_score = 60.0
        else:
            peg_score = max(0.0, 100 - (peg - 2.0) * 20)

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

    quality_score = 100 * margin_penalty * debt_penalty
    score = round(s_yoy * 0.35 + s_val * 0.40 + peg_score * 0.15 + quality_score * 0.10, 1)
    details = {
        'yoy': yoy,
        'mom_turned_positive': bool(mom_turned_positive),
        'per_current': float(current_per) if current_per is not None else None,
        'per_percentile': percentile,
        'peg_score': peg_score,
        'margin_penalty': margin_penalty,
        'debt_penalty': debt_penalty,
    }
    return score, details

def score_chip(foreign_net, trust_net, volume, proprietary_net=None):
    """P0 優化：連續買超天數上限改 10 天（更符合現實）、加入買超加速度指標、
    自營商資料（若抓得到）作為反向訊號警示。"""
    def streak_count(net_series):
        if net_series is None or net_series.empty:
            return 0
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
    p_streak = streak_count(proprietary_net)
    s_foreign = min(100.0, f_streak / 10 * 100)
    s_trust = min(100.0, t_streak / 10 * 100)
    s_proprietary = min(100.0, p_streak / 10 * 100) if proprietary_net is not None and not proprietary_net.empty else 50.0

    acceleration_bonus = 0.0
    recent_5d = foreign_net.tail(5)
    if len(recent_5d) >= 3:
        trend = recent_5d.iloc[-1] - recent_5d.iloc[-3]
        if trend > 0:
            acceleration_bonus = 10.0

    combined_5d = foreign_net.tail(5).sum() + trust_net.tail(5).sum()
    volume_5d = volume.tail(5).sum()
    ratio = combined_5d / (volume_5d + 1e-6)  # 用股數比例代替金額比例（FinMind 籌碼資料為股數，非金額）
    if ratio > 0.05:
        s_strength = 100.0
    elif ratio > 0.02:
        s_strength = min(100.0, ratio / 0.05 * 100)
    else:
        s_strength = max(0.0, ratio / 0.02 * 50)

    prop_vs_inst = 1.0
    prop_warning = False
    if proprietary_net is not None and not proprietary_net.empty:
        prop_recent = proprietary_net.tail(5).sum()
        inst_recent = foreign_net.tail(5).sum() + trust_net.tail(5).sum()
        if prop_recent < -100 and inst_recent > 100:
            prop_vs_inst = 0.85
            prop_warning = True

    score = round(
        (s_foreign * 0.25 + s_trust * 0.35 + s_proprietary * 0.15 + s_strength * 0.25 + acceleration_bonus)
        * prop_vs_inst, 1
    )
    details = {
        'foreign_streak': f_streak,
        'trust_streak': t_streak,
        'proprietary_streak': p_streak,
        'strength_ratio_pct': round(ratio * 100, 2),
        'acceleration_bonus': acceleration_bonus,
        'prop_vs_inst_warning': prop_warning,
    }
    return score, details

def build_stock_report(name, stock_id, latest_price, tech_score, tech_d,
                        fund_score, fund_d, chip_score, chip_d,
                        macro_score, macro_d, total_score, passed, score_threshold):
    yoy_txt = f"{fund_d['yoy']:.1f}%" if fund_d['yoy'] is not None else "無資料"
    per_txt = f"{fund_d['per_current']:.1f} 倍" if fund_d['per_current'] is not None else "無資料"
    per_pct_txt = f"（近2年百分位 {fund_d['per_percentile']:.0f}%，越低越便宜）" if fund_d['per_percentile'] is not None else ""
    mom_txt = "，月營收由負轉正" if fund_d['mom_turned_positive'] else ""
    quality_txt = ""
    if fund_d.get('margin_penalty', 1.0) < 1.0 or fund_d.get('debt_penalty', 1.0) < 1.0:
        quality_txt = "（⚠️ 毛利率或負債比未達標，基本面已打折）"

    accel_txt = "，買超力道正在加速" if chip_d.get('acceleration_bonus', 0) > 0 else ""
    prop_txt = "｜⚠️ 自營商同期賣超，與法人方向不一致" if chip_d.get('prop_vs_inst_warning') else ""

    us_txt = f"{macro_d['us_return_pct']:+.2f}%" if macro_d.get('us_return_pct') is not None else "無資料"
    jp_txt = f"{macro_d['jp_return_pct']:+.2f}%" if macro_d.get('jp_return_pct') is not None else "無資料"

    lines = [
        f"【{name}({stock_id})】最新收盤 NT$ {latest_price}",
        f"總分 {total_score} 分（進場門檻 {score_threshold} 分）— {'✅ 達進場標準' if passed else '⚪ 尚未達標，列入觀察'}",
        f"技術面 {tech_score} 分：{tech_d['trend_label']}，乖離率 {tech_d['bias_pct']}%，"
        f"量比 {tech_d['vol_ratio']} 倍（{'收紅' if tech_d['is_red'] else '收黑'}），"
        f"動能百分位（短60/長252天融合）{tech_d['momentum_pct']}%",
        f"基本面 {fund_score} 分：營收 YoY {yoy_txt}{mom_txt}，PE {per_txt}{per_pct_txt}{quality_txt}",
        f"籌碼面 {chip_score} 分：外資連續買超 {chip_d['foreign_streak']} 天，"
        f"投信連續買超 {chip_d['trust_streak']} 天，"
        f"近5日法人買超力道占成交量比例 {chip_d['strength_ratio_pct']}%{accel_txt}{prop_txt}",
        f"跨市場 {macro_score} 分：美股(S&P500/那斯達克平均) {us_txt} ｜ 日股(日經225) {jp_txt}（全市場共用同一組數值）",
        f"成交量：今日 {tech_d['volume_today']:,.0f} 股，20日均量 {tech_d['volume_avg20']:,.0f} 股，"
        f"較均量{'放大' if tech_d['volume_surge_pct'] >= 0 else '萎縮'} {abs(tech_d['volume_surge_pct']):.1f}%",
    ]
    return "\n".join(lines)

def calculate_stops(close_price, atr14, volatility_regime='NORMAL', beta=DEFAULT_BETA):
    """P0 優化：停損停利倍數依大盤波動率動態調整、依個股 Beta 加寬（beta=1.0 時無調整）、
    強制最低風報比 1.5:1（不足則放寬停利，不動停損以免風險擴大）。"""
    if volatility_regime == 'LOW':
        k_sl_base, k_tp_base = 1.5, 2.5
    elif volatility_regime == 'HIGH':
        k_sl_base, k_tp_base = 2.0, 3.5
    else:
        k_sl_base, k_tp_base = K_SL, K_TP

    # beta=1.0（無資料時的預設值）時 beta_factor=0，不做任何調整；beta>1 時停損停利同步加寬
    beta_factor = max(0.0, (beta - 1.0) * 0.3)
    k_sl = k_sl_base + beta_factor
    k_tp = k_tp_base + beta_factor * 0.5

    sl_price = close_price - k_sl * atr14
    tp_price = close_price + k_tp * atr14
    if sl_price <= 0:
        sl_price = max(0.01, close_price * 0.8)

    risk_amt = close_price - sl_price
    profit_amt = tp_price - close_price
    rr_ratio = profit_amt / (risk_amt + 1e-6)
    if rr_ratio < 1.5:
        tp_price = close_price + (close_price - sl_price) * 1.5
        rr_ratio = 1.5

    return {
        'sl_price': round(sl_price, 2), 'tp_price': round(tp_price, 2),
        'rr_ratio': round(rr_ratio, 2), 'k_sl': round(k_sl, 3), 'k_tp': round(k_tp, 3),
    }

def calculate_position_size(buy_price, sl_price, risk_pct):
    """P0 優化：加入邊界檢查（停損高於買價、停損幅度過大直接不進場），部位四捨五入到 0.5% 級距。"""
    if buy_price <= 0 or sl_price >= buy_price:
        return 0.0
    sl_pct = (buy_price - sl_price) / buy_price
    if sl_pct <= 0 or sl_pct > 0.5:
        return 0.0
    position_size = min(risk_pct / sl_pct, MAX_POSITION_PCT)
    return round(position_size * 2) / 2

def check_entry_conditions(tech_score, fund_score, chip_score, total_score,
                            score_threshold, daily_volume_amt=None, beta=DEFAULT_BETA):
    """P0 優化風控：單項因子門檻 40→50、加入反向訊號判斷（技術破位但基本籌碼強→棄權等確認）、
    流動性檢查（日成交金額 <1億排除）、高 Beta 股更嚴格門檻（beta=1.0 預設值下此條件不生效）。"""
    threshold = max(score_threshold, 75.0) if beta > 1.3 else score_threshold
    if total_score < threshold:
        return False, f"總分不足 ({total_score} < {threshold})"

    min_score = min(tech_score, fund_score, chip_score)
    if min_score < MIN_FACTOR_SCORE:
        return False, f"存在弱因子 (最低: {min_score} < {MIN_FACTOR_SCORE})"

    if tech_score < 45 and (fund_score + chip_score) > 140:
        return False, "技術面疲弱但基本籌碼面強，棄權等待技術面確認"

    if daily_volume_amt is not None and daily_volume_amt < MIN_DAILY_TURNOVER:
        return False, f"日成交金額太小 ({daily_volume_amt/1e8:.2f}億 < 1億)，易滑點"

    if beta > 1.5 and (total_score < 75 or min_score < 55):
        return False, f"高 Beta ({beta}) 需更高門檻"

    return True, f"✅ 進場確認 (總分: {total_score})"

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 啟動 AI 多因子選股系統...")
    audit_past_predictions()

    adapted_threshold, historical_win_rate = compute_self_training_metrics()
    if historical_win_rate is not None:
        print(f"📊 [自我訓練] 累積歷史勝率 {historical_win_rate*100:.1f}%"
              + (f"，門檻自動收緊至 {adapted_threshold} 分" if adapted_threshold else "，暫不調整門檻") + "\n")
    else:
        print(f"📊 [自我訓練] 已驗證樣本數不足 {SELF_TRAIN_MIN_SAMPLES} 筆，暫不調整門檻，沿用固定值\n")

    market_regime, volatility_regime, _ = detect_market_regime()
    if market_regime == 'BEAR':
        score_threshold = BEAR_SCORE_THRESHOLD
        risk_pct = RISK_PER_TRADE_PCT / 2
    else:
        score_threshold = ENTRY_SCORE_THRESHOLD
        risk_pct = RISK_PER_TRADE_PCT
    if adapted_threshold and adapted_threshold > score_threshold:
        score_threshold = adapted_threshold  # 自我訓練只會收緊門檻，不會放寬（見 compute_self_training_metrics 的說明）
    print(f"🎯 今日進場總分門檻: {score_threshold} 分 ｜ 單筆風險: {risk_pct}% 總資金\n")

    dynamic_stock_pool = get_market_active_stocks()
    print(f"✅ 鎖定 {len(dynamic_stock_pool)} 支熱門標的，開始執行多因子評分...\n")

    print("📡 抓取近期全市場三大法人買賣超（TWSE T86，一天一次呼叫，取代原本一股一次的 FinMind 呼叫）...")
    inst_data_by_date = fetch_recent_institutional_data()
    print(f"✅ 取得 {len(inst_data_by_date)} 個交易日的法人資料。\n")

    print("📡 抓取最新一期全市場月營收（MOPS 公開資訊觀測站，2 次請求涵蓋全市場，取代逐股呼叫 FinMind）...")
    mops_revenue_data = fetch_mops_revenue_snapshot()
    print(f"✅ 取得 {len(mops_revenue_data)} 家公司的月營收資料。\n")

    print("📡 抓取跨市場領先指標（美股 S&P500/那斯達克、日股日經225，yfinance，全市場共用單一訊號）...")
    us_return_pct, jp_return_pct = fetch_cross_market_signal()
    macro_score, macro_detail = score_macro(us_return_pct, jp_return_pct)
    print(f"✅ 跨市場因子分數: {macro_score}（美股 {us_return_pct}% ｜ 日股 {jp_return_pct}%）\n")

    today_str = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')      # 動能百分位需要約 1 年資料
    per_start_date = (datetime.now() - timedelta(days=1150)).strftime('%Y-%m-%d')  # 估值河流位階需要約 3 年資料

    init_db()
    conn = sqlite3.connect(DB_NAME)
    results = []
    all_candidates = []

    empty_price_count = 0
    for stock_id, name in dynamic_stock_pool.items():
        try:
            df = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date)
            time.sleep(0.15)
            if df.empty or len(df) < 60:
                empty_price_count += 1
                continue
            df = df.rename(columns={'max': 'High', 'min': 'Low', 'close': 'Close',
                                     'open': 'Open', 'Trading_Volume': 'Volume'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            # 籌碼面：三大法人買賣超（含自營商），改用 TWSE T86 全市場資料（main() 迴圈外已一次性
            # 抓好 inst_data_by_date），不用再逐股呼叫 FinMind，自營商欄位也是官方正確欄位，不用再用猜的。
            # 這三個 Series 是各自獨立按日期排序的（tail(n) 語意），不需要也不能對齊 df 的價格列數。
            foreign_series, trust_series, dealer_series = build_institutional_series(inst_data_by_date, stock_id)

            # 基本面：月營收 YoY / MoM 轉正，改用 MOPS 全市場快照（main() 迴圈外已一次性抓好 mops_revenue_data），
            # 不用再逐股呼叫 FinMind；MOPS 回傳的 yoy_pct 已經是正確的百分比數值，不需要再做 *100 的猜測性修正。
            mops_rec = mops_revenue_data.get(stock_id)
            latest_yoy = mops_rec['yoy_pct'] if mops_rec else None
            mom_turned_positive = mops_rec['mom_turned_positive'] if mops_rec else False

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

            tech_score, tech_detail = score_technical(df, volatility_regime=volatility_regime)
            if tech_score is None:
                continue
            fund_score, fund_detail = score_fundamental(latest_yoy, mom_turned_positive, per_series)
            chip_score, chip_detail = score_chip(foreign_series, trust_series, df['Volume'],
                                                  proprietary_net=dealer_series)

            total_score = round(
                tech_score * WEIGHT_TECH + fund_score * WEIGHT_FUND +
                chip_score * WEIGHT_CHIP + macro_score * WEIGHT_MACRO, 1
            )

            daily_volume_amt = float(df['Trading_money'].iloc[-1]) if 'Trading_money' in df.columns else None
            passed, entry_reason = check_entry_conditions(
                tech_score, fund_score, chip_score, total_score, score_threshold,
                daily_volume_amt=daily_volume_amt, beta=DEFAULT_BETA
            )

            latest_price = round(float(df['Close'].iloc[-1]), 2)
            atr_series = get_atr14(df)
            latest_atr = atr_series.iloc[-1] if pd.notna(atr_series.iloc[-1]) else latest_price * 0.02

            buy_price = latest_price
            stops = calculate_stops(buy_price, latest_atr, volatility_regime=volatility_regime, beta=DEFAULT_BETA)
            sl_price = stops['sl_price']
            tp_price = stops['tp_price']
            pos_size = calculate_position_size(buy_price, sl_price, risk_pct)

            status_label = "🔥 建議買進" if passed else f"☁️ 觀望（{entry_reason}）"

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
                         trailing_stop_price, entry_atr, tech_score, fund_score, chip_score, macro_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (today_str, stock_id, name, latest_price, buy_price, tp_price, sl_price,
                          total_score,
                          round(latest_yoy, 1) if latest_yoy is not None else None,
                          round(latest_per, 1) if latest_per is not None else None,
                          pos_size, market_regime, sl_price, round(float(latest_atr), 4),
                          tech_score, fund_score, chip_score, macro_score))

            results.append({
                '股票代碼': stock_id, '股票名稱': name, '最新收盤價': latest_price,
                '技術面': tech_score, '基本面': fund_score, '籌碼面': chip_score, '跨市場': macro_score,
                '總分': total_score, '決策建議': status_label,
                '建議買入價': buy_price, '停利價': tp_price, '停損價': sl_price,
                '建議部位(%)': pos_size
            })

            report_text = build_stock_report(
                name, stock_id, latest_price, tech_score, tech_detail,
                fund_score, fund_detail, chip_score, chip_detail,
                macro_score, macro_detail, total_score, passed, score_threshold
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

    print(f"📊 掃描完成：{len(all_candidates)}/{len(dynamic_stock_pool)} 支成功取得評分，"
          f"{empty_price_count} 支股價資料為空或不足 60 筆（可能是 FinMind 額度用盡或當日尚無資料）。")

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
    ensure_finmind_login()
    main()
