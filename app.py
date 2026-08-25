import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 1. 頁面基礎設定
st.set_page_config(page_title="台股 AI 量化智庫與互動終端", page_icon="📈", layout="wide")

# 第三方套件載入防護
HAS_PLOTLY = False
try:
    import plotly.graph_objects as pgo
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    pass

HAS_FINMIND = False
try:
    from FinMind.data import DataLoader
    HAS_FINMIND = True
except Exception:
    pass

DB_NAME = "paper_trading.db"

# --- 修正 1：Token 不再寫死在程式碼中 ---
# 優先順序：st.secrets > 環境變數 > 側邊欄手動輸入
def get_token():
    try:
        if "FINMIND_TOKEN" in st.secrets:
            return st.secrets["FINMIND_TOKEN"]
    except Exception:
        pass
    env_token = os.environ.get("FINMIND_TOKEN")
    if env_token:
        return env_token
    return st.sidebar.text_input("FinMind API Token（未設定 secrets 時手動輸入）", type="password")

FINMIND_TOKEN = get_token()

# --- 修正 2：登入結果不再靜默吞掉，明確顯示連線狀態 ---
def get_finmind_loader(token):
    if not HAS_FINMIND:
        st.sidebar.warning("⚠️ 尚未安裝 FinMind 套件（pip install FinMind），個股資料功能將無法使用。")
        return None
    if not token:
        st.sidebar.info("ℹ️ 尚未提供 FinMind Token，個股資料功能將無法使用。")
        return None
    try:
        loader = DataLoader()
        loader.login_by_token(token=token)
        st.sidebar.success("✅ FinMind 資料源連線成功")
        return loader
    except Exception as e:
        st.sidebar.error(f"❌ FinMind 登入失敗：{e}")
        return None

dl = get_finmind_loader(FINMIND_TOKEN)

# 2. 資料庫初始化（保留原本結構，但錯誤會顯示出來）
def init_all_tables():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predict_date TEXT, stock_id TEXT, stock_name TEXT,
                latest_price REAL, buy_price REAL, tp_price REAL, sl_price REAL,
                ai_win_rate REAL, status TEXT DEFAULT 'PENDING',
                real_max_price REAL DEFAULT 0, real_min_price REAL DEFAULT 0,
                validated_date TEXT, revenue_yoy REAL, pe_ratio REAL,
                position_size REAL DEFAULT 0.0, market_regime TEXT DEFAULT 'NORMAL'
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                stock_id TEXT PRIMARY KEY, stock_name TEXT, added_date TEXT
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"資料庫初始化失敗：{e}")

init_all_tables()

# --- 修正 3：predictions 表格原本永遠沒有資料來源，這裡補一個「示範資料」產生器 ---
# 這不是真正的 AI 選股邏輯，只是讓介面有東西可以顯示。
# 你要接自己的模型時，把真正的推薦結果 INSERT 進 predictions 表即可，
# 欄位格式跟這個函式一致。
def seed_demo_predictions_if_empty():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        today_str = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE predict_date = ?", (today_str,))
        count_today = cursor.fetchone()[0]
        if count_today == 0:
            demo_rows = [
                (today_str, "2330", "台積電", 1050.0, 1045.0, 1120.0, 1010.0, 68.5, "PENDING", 0, 0, None, 12.3, 22.1, 8.0, "NORMAL"),
                (today_str, "2317", "鴻海", 205.0, 203.0, 220.0, 195.0, 61.2, "PENDING", 0, 0, None, 5.6, 11.4, 5.0, "NORMAL"),
            ]
            cursor.executemany('''
                INSERT INTO predictions
                (predict_date, stock_id, stock_name, latest_price, buy_price, tp_price, sl_price,
                 ai_win_rate, status, real_max_price, real_min_price, validated_date,
                 revenue_yoy, pe_ratio, position_size, market_regime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', demo_rows)
            conn.commit()
        conn.close()
    except Exception as e:
        st.warning(f"示範資料建立失敗（不影響其他功能）：{e}")

seed_demo_predictions_if_empty()

# 3. 自選股安全操作函數
def add_to_watchlist(stock_id):
    stock_id = str(stock_id).strip()
    if not stock_id:
        return False, "請輸入有效的股票代碼！"

    stock_name = f"股票 {stock_id}"
    if dl:
        try:
            info = dl.taiwan_stock_info()
            if not info.empty and 'stock_id' in info.columns:
                matched = info[info['stock_id'] == stock_id]
                if not matched.empty:
                    stock_name = str(matched['stock_name'].iloc[0])
        except Exception:
            pass

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        today_str = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("INSERT INTO watchlist (stock_id, stock_name, added_date) VALUES (?, ?, ?)",
                       (stock_id, stock_name, today_str))
        conn.commit()
        conn.close()
        return True, f"成功加入自選股：{stock_name} ({stock_id})"
    except Exception:
        return False, "加入失敗（該標的可能已在自選股清單中）"

def remove_from_watchlist(stock_id):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM watchlist WHERE stock_id=?", (stock_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"刪除失敗：{e}")

def get_watchlist():
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql("SELECT * FROM watchlist", conn)
        conn.close()
        return df if not df.empty else pd.DataFrame(columns=['stock_id', 'stock_name', 'added_date'])
    except Exception:
        return pd.DataFrame(columns=['stock_id', 'stock_name', 'added_date'])

# 4. 側邊欄渲染
st.sidebar.title("⭐ 個人自選股清單")

with st.sidebar.form("add_stock_form", clear_on_submit=True):
    new_stock_id = st.text_input("輸入股票代碼 (例: 2330)", "")
    submit_add = st.form_submit_button("➕ 新增至關注清單")
    if submit_add:
        success, msg = add_to_watchlist(new_stock_id)
        if success:
            st.sidebar.success(msg)
            st.rerun()
        else:
            st.sidebar.error(msg)

df_wl = get_watchlist()
if not df_wl.empty and 'stock_id' in df_wl.columns:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 當前關注標的")
    for _, wl_row in df_wl.iterrows():
        s_id = str(wl_row['stock_id'])
        s_name = str(wl_row.get('stock_name', s_id))
        col_wl1, col_wl2 = st.sidebar.columns([3, 1])
        col_wl1.write(f"**{s_name}** ({s_id})")
        if col_wl2.button("❌", key=f"del_{s_id}"):
            remove_from_watchlist(s_id)
            st.rerun()
else:
    st.sidebar.info("關注清單為空，請於上方輸入代碼新增。")

# 5. 主頁面標題與數據載入
st.title("📈 台股 AI 量化智庫與互動視覺化終端")
st.caption("結合 AI 選股模型、個人自選股清單、動態 K 線圖與法人籌碼/財報圖解")

def load_predictions():
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql("SELECT * FROM predictions", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"讀取歷史預測失敗：{e}")
        return pd.DataFrame()

df_all = load_predictions()

total_cnt = 0
win_cnt = 0
win_rate = 0.0

if not df_all.empty and 'status' in df_all.columns:
    completed = df_all[df_all['status'] != 'PENDING']
    wins = df_all[df_all['status'] == 'WIN (成功停利)']
    total_cnt = len(completed)
    win_cnt = len(wins)
    win_rate = (win_cnt / total_cnt * 100) if total_cnt > 0 else 0.0

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("歷史總推薦單數", f"{len(df_all)} 筆")
with col2:
    st.metric("已結算驗證單數", f"{total_cnt} 筆")
with col3:
    st.metric("真實實戰勝率", f"{win_rate:.1f}%")

st.divider()

# 6. 頁籤渲染
tab1, tab2, tab3 = st.tabs(["🔥 今日 AI 精選決策", "🔍 個股 K 線 / 籌碼 / 財報深度圖解", "📜 歷史預測對照表"])

today_str = datetime.now().strftime('%Y-%m-%d')
df_today = pd.DataFrame()
if not df_all.empty and 'predict_date' in df_all.columns:
    df_today = df_all[df_all['predict_date'] == today_str]

# Tab 1: 今日 AI 精選
with tab1:
    st.subheader(f"🤖 今日 ({today_str}) AI 精選標的與建議")
    st.caption("⚠️ 目前顯示為示範資料，尚未接上真正的 AI 選股引擎（見程式中 seed_demo_predictions_if_empty）。")
    if not df_today.empty:
        for _, row in df_today.iterrows():
            l_price = float(row.get('latest_price', 0))
            tp_p = float(row.get('tp_price', 0))
            sl_p = float(row.get('sl_price', 0))
            upside = round(((tp_p - l_price) / (l_price + 1e-6)) * 100, 1)
            downside = round(((l_price - sl_p) / (l_price + 1e-6)) * 100, 1)
            rr_ratio = round(upside / (downside + 1e-6), 2)
            rev_yoy_raw = row.get('revenue_yoy', None)
            rev_str = f"{float(rev_yoy_raw):.1f}%" if pd.notna(rev_yoy_raw) and rev_yoy_raw != 'N/A' else "未提供"
            pos_size = row.get('position_size', 5.0)

            st.markdown(f"""
            <div style="background:#ffffff; border-left:5px solid #2563eb; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:15px;">
                <h3 style="margin:0; color:#1e3a8a;">📌 {row.get('stock_name', '未知')} ({row.get('stock_id', '')})</h3>
                <p style="margin-top:5px; color:#475569;">
                    <b>最新收盤價：</b> NT$ {l_price} ｜
                    <b>AI 勝率：</b> <span style="color:#d97706; font-size:18px; font-weight:bold;">{row.get('ai_win_rate', 0)}%</span> ｜
                    <b>建議部位：</b> <span style="color:#2563eb; font-size:18px; font-weight:bold;">{pos_size}% 總資金</span> ｜
                    <b>營收 YoY：</b> <span style="color:#16a34a; font-weight:bold;">{rev_str}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("📖 點擊查看風控細節與買賣點算價"):
                st.write(f"- **建議買入價**：`NT$ {row.get('buy_price', l_price)}` | **停利價**：`NT$ {tp_p}` | **停損價**：`NT$ {sl_p}` | **風報比**：`{rr_ratio}`")
    else:
        st.info("今日市場經 AI 與風控過濾後無符合進場條件之標的，建議觀望保持現金。")

# Tab 2: K線與圖表
with tab2:
    st.subheader("📊 互動式技術面、籌碼面與基本面圖解")

    if not HAS_PLOTLY:
        st.warning("⚠️ 尚未安裝 plotly（pip install plotly），將以簡易折線圖代替 K 線圖。")
    if not dl:
        st.warning("⚠️ FinMind 未連線成功，此分頁功能暫時無法使用（請檢查側邊欄的連線狀態訊息）。")

    stock_options = {}
    if not df_today.empty:
        for _, r in df_today.iterrows():
            stock_options[str(r['stock_id'])] = f"[AI 推薦] {r.get('stock_name', '')} ({r['stock_id']})"
    if not df_wl.empty and 'stock_id' in df_wl.columns:
        for _, r in df_wl.iterrows():
            sid = str(r['stock_id'])
            if sid not in stock_options:
                stock_options[sid] = f"[關注清單] {r.get('stock_name', sid)} ({sid})"
    if not stock_options:
        stock_options['2330'] = "[熱門預設] 台積電 (2330)"

    selected_stock_id = st.selectbox("請選擇欲分析診斷的股票：", options=list(stock_options.keys()), format_func=lambda x: stock_options[x])

    if selected_stock_id and dl:
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        df_stock, df_chip, df_rev, df_per = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        with st.spinner(f"載入 {selected_stock_id} 數據中..."):
            try:
                df_stock = dl.taiwan_stock_daily(stock_id=selected_stock_id, start_date=start_date)
            except Exception as e:
                st.error(f"抓取股價失敗：{e}")
            try:
                df_chip = dl.taiwan_stock_institutional_investors(stock_id=selected_stock_id, start_date=start_date)
            except Exception:
                pass
            try:
                df_rev = dl.taiwan_stock_month_revenue(stock_id=selected_stock_id, start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
            except Exception:
                pass
            try:
                df_per = dl.taiwan_stock_per_pbr(stock_id=selected_stock_id, start_date=start_date)
            except Exception:
                pass

        if not df_stock.empty and 'close' in df_stock.columns and len(df_stock) > 5:
            try:
                df_stock = df_stock.rename(columns={'max': 'High', 'min': 'Low', 'close': 'Close', 'open': 'Open', 'Trading_Volume': 'Volume'})
                df_stock['date'] = pd.to_datetime(df_stock['date'])
                df_stock = df_stock.sort_values('date').reset_index(drop=True)

                df_stock['MA5'] = df_stock['Close'].rolling(5).mean()
                df_stock['MA20'] = df_stock['Close'].rolling(20).mean()

                foreign_net = pd.Series(0, index=df_stock.index)
                trust_net = pd.Series(0, index=df_stock.index)
                if not df_chip.empty and 'name' in df_chip.columns:
                    df_chip['date'] = pd.to_datetime(df_chip['date'])
                    f_buy = df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['buy'].sum() - df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['sell'].sum()
                    t_buy = df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['buy'].sum() - df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['sell'].sum()
                    foreign_net = df_stock['date'].map(f_buy).fillna(0) / 1000.0
                    trust_net = df_stock['date'].map(t_buy).fillna(0) / 1000.0

                if HAS_PLOTLY:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35],
                                        subplot_titles=(f"{selected_stock_id} K 線圖 (MA5 / MA20)", "三大法人買賣超 (張)"))

                    fig.add_trace(pgo.Candlestick(
                        x=df_stock['date'], open=df_stock['Open'], high=df_stock['High'], low=df_stock['Low'], close=df_stock['Close'],
                        name="日 K 線", increasing_line_color='#ef4444', decreasing_line_color='#22c55e'
                    ), row=1, col=1)

                    fig.add_trace(pgo.Scatter(x=df_stock['date'], y=df_stock['MA5'], mode='lines', name='MA5', line=dict(color='#f59e0b', width=1.5)), row=1, col=1)
                    fig.add_trace(pgo.Scatter(x=df_stock['date'], y=df_stock['MA20'], mode='lines', name='MA20', line=dict(color='#3b82f6', width=1.5)), row=1, col=1)

                    fig.add_trace(pgo.Bar(x=df_stock['date'], y=foreign_net, name='外資(張)', marker_color='#8b5cf6'), row=2, col=1)
                    fig.add_trace(pgo.Bar(x=df_stock['date'], y=trust_net, name='投信(張)', marker_color='#ec4899'), row=2, col=1)

                    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart(df_stock.set_index('date')[['Close', 'MA5', 'MA20']])

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.write("**💰 估值指標 (PE / PB)**")
                    if not df_per.empty and 'PER' in df_per.columns:
                        st.info(f"- 本益比 (PE): `{df_per['PER'].iloc[-1]} 倍`\n- 股價淨值比 (PB): `{df_per['PBR'].iloc[-1]} 倍`")
                    else:
                        st.write("無估值數據。")
                with col_f2:
                    st.write("**📊 近期月營收**")
                    if not df_rev.empty and 'revenue' in df_rev.columns:
                        cols_to_show = [c for c in ['revenue_year', 'revenue_month', 'revenue'] if c in df_rev.columns]
                        st.dataframe(df_rev.tail(5)[cols_to_show], hide_index=True)
                    else:
                        st.write("無月營收數據。")
            except Exception as e:
                st.warning(f"圖表繪製遭遇異常，已降級顯示數據: {e}")
        else:
            st.info(f"目前抓不到 {selected_stock_id} 的股價資料（可能是代碼錯誤、FinMind 額度用盡，或該股近期無交易資料）。")

# Tab 3
with tab3:
    st.subheader("📜 歷史預測紀錄與實戰對照明細")
    if not df_all.empty:
        st.dataframe(df_all.sort_values(by='id', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("無歷史預測紀錄。")
