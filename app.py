import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. 頁面配置
st.set_page_config(page_title="台股 AI 量化智庫與互動終端", page_icon="📈", layout="wide")

# 動態載入第三方庫防護
try:
    import plotly.graph_objects as pgo
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    from FinMind.data import DataLoader
    HAS_FINMIND = True
except Exception:
    HAS_FINMIND = False

DB_NAME = "paper_trading.db"
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiMDUxOGNoaXl1QGdtYWlsLmNvbSIsImVtYWlsIjoiMDUxOGNoaXl1QGdtYWlsLmNvbSIsInRva2VuX3ZlcnNpb24iOjAsImV4cCI6MTc4ODI0MDUwOH0.dNGO-ZUPpWW30mfiUdwMqIJV-v2bqShtiLJsoy4vh7I"

# 安全初始化 DataLoader (移除會對 None 建立弱引用的 @st.cache_resource 裝飾器)
def get_finmind_loader():
    if not HAS_FINMIND:
        return None
    try:
        loader = DataLoader()
        loader.login_by_token(token=FINMIND_TOKEN)
        return loader
    except Exception:
        return None

dl = get_finmind_loader()

# 2. 資料庫安全初始化
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
        st.warning(f"資料庫初始化提示: {e}")

init_all_tables()

def add_to_watchlist(stock_id):
    stock_id = stock_id.strip()
    if not stock_id:
        return False, "請輸入股票代碼！"
    
    stock_name = f"股票 {stock_id}"
    if dl:
        try:
            info = dl.taiwan_stock_info()
            matched = info[info['stock_id'] == stock_id]
            if not matched.empty:
                stock_name = matched['stock_name'].iloc[0]
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
    except Exception as e:
        return False, f"加入失敗 (可能已存在): {e}"

def remove_from_watchlist(stock_id):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM watchlist WHERE stock_id=?", (stock_id,))
        conn.commit()
        conn.close()
    except Exception:
        pass

def get_watchlist():
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql("SELECT * FROM watchlist", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# 3. 側邊欄：自選股關注清單管理
st.sidebar.title("⭐ 個人自選股清單")
with st.sidebar.form("add_stock_form", clear_on_submit=True):
    new_stock_id = st.text_input("輸入股票代碼 (例: 2330)", "")
    submit_add = st.form_submit_button("➕ 新增至關注清單")
    if submit_add:
        success, msg = add_to_watchlist(new_stock_id)
        if success:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

df_wl = get_watchlist()
if not df_wl.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 當前關注標的")
    for _, wl_row in df_wl.iterrows():
        col_wl1, col_wl2 = st.sidebar.columns([3, 1])
        col_wl1.write(f"**{wl_row['stock_name']}** ({wl_row['stock_id']})")
        if col_wl2.button("❌", key=f"del_{wl_row['stock_id']}"):
            remove_from_watchlist(wl_row['stock_id'])
            st.rerun()
else:
    st.sidebar.info("關注清單為空，請輸入代碼新增股票。")

# 4. 主頁面標題與數據載入
st.title("📈 台股 AI 量化智庫與互動視覺化終端")
st.caption("結合 AI 選股模型、個人自選股清單、動態 K 線圖與法人籌碼/財報圖解")

def load_predictions():
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql("SELECT * FROM predictions", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

df_all = load_predictions()
completed = df_all[df_all['status'] != 'PENDING'] if not df_all.empty and 'status' in df_all.columns else pd.DataFrame()
wins = df_all[df_all['status'] == 'WIN (成功停利)'] if not df_all.empty and 'status' in df_all.columns else pd.DataFrame()

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

# 5. 頁籤分類
tab1, tab2, tab3 = st.tabs(["🔥 今日 AI 精選決策", "🔍 個股 K 線 / 籌碼 / 財報深度圖解", "📜 歷史預測對照表"])

today_str = datetime.now().strftime('%Y-%m-%d')
df_today = df_all[df_all['predict_date'] == today_str] if not df_all.empty and 'predict_date' in df_all.columns else pd.DataFrame()

# Tab 1: 今日 AI 精選
with tab1:
    st.subheader(f"🤖 今日 ({today_str}) AI 精選標的與建議")
    if not df_today.empty:
        for _, row in df_today.iterrows():
            upside = round(((row['tp_price'] - row['latest_price']) / row['latest_price']) * 100, 1)
            downside = round(((row['latest_price'] - row['sl_price']) / row['latest_price']) * 100, 1)
            rr_ratio = round(upside / (downside + 1e-6), 2)
            rev_yoy_raw = row.get('revenue_yoy', None)
            rev_str = f"{float(rev_yoy_raw):.1f}%" if pd.notna(rev_yoy_raw) and rev_yoy_raw != 'N/A' else "未提供"
            pos_size = row.get('position_size', 5.0)

            st.markdown(f"""
            <div style="background:#ffffff; border-left:5px solid #2563eb; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin-bottom:15px;">
                <h3 style="margin:0; color:#1e3a8a;">📌 {row['stock_name']} ({row['stock_id']})</h3>
                <p style="margin-top:5px; color:#475569;">
                    <b>最新收盤價：</b> NT$ {row['latest_price']} ｜ 
                    <b>AI 勝率：</b> <span style="color:#d97706; font-size:18px; font-weight:bold;">{row['ai_win_rate']}%</span> ｜
                    <b>建議部位：</b> <span style="color:#2563eb; font-size:18px; font-weight:bold;">{pos_size}% 總資金</span> ｜
                    <b>營收 YoY：</b> <span style="color:#16a34a; font-weight:bold;">{rev_str}</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
            with st.expander(f"📖 點擊查看 {row['stock_name']} 風控細節與買賣點算價"):
                st.write(f"- **建議買入價**：`NT$ {row['buy_price']}` | **停利價**：`NT$ {row['tp_price']}` | **停損價**：`NT$ {row['sl_price']}`")
    else:
        st.info("今日市場經 AI 與風控過濾後無符合進場條件之標的，建議觀望保持現金。")

# Tab 2: 視覺化圖表與財報深度分析
with tab2:
    st.subheader("📊 互動式技術面、籌碼面與基本面圖解")
    stock_options = {}
    if not df_today.empty:
        for _, r in df_today.iterrows():
            stock_options[r['stock_id']] = f"[AI 推薦] {r['stock_name']} ({r['stock_id']})"
    if not df_wl.empty:
        for _, r in df_wl.iterrows():
            if r['stock_id'] not in stock_options:
                stock_options[r['stock_id']] = f"[關注清單] {r['stock_name']} ({r['stock_id']})"
    if not stock_options:
        stock_options['2330'] = "[熱門預設] 台積電 (2330)"

    selected_stock_id = st.selectbox("請選擇欲分析診斷的股票：", options=list(stock_options.keys()), format_func=lambda x: stock_options[x])

    if selected_stock_id and HAS_PLOTLY and dl:
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        with st.spinner(f"載入 {selected_stock_id} 數據中..."):
            try:
                df_stock = dl.taiwan_stock_daily(stock_id=selected_stock_id, start_date=start_date)
                df_chip = dl.taiwan_stock_institutional_investors(stock_id=selected_stock_id, start_date=start_date)
                df_rev = dl.taiwan_stock_month_revenue(stock_id=selected_stock_id, start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
                df_per = dl.taiwan_stock_per_pbr(stock_id=selected_stock_id, start_date=start_date)
            except Exception as e:
                st.error(f"數據抓取失敗: {e}")
                df_stock = pd.DataFrame()

        if not df_stock.empty and 'close' in df_stock.columns:
            df_stock = df_stock.rename(columns={'max': 'High', 'min': 'Low', 'close': 'Close', 'open': 'Open', 'Trading_Volume': 'Volume'})
            df_stock['date'] = pd.to_datetime(df_stock['date'])
            df_stock = df_stock.sort_values('date').reset_index(drop=True)

            df_stock['MA5'] = df_stock['Close'].rolling(5).mean()
            df_stock['MA20'] = df_stock['Close'].rolling(20).mean()

            foreign_net = pd.Series(0, index=df_stock.index)
            trust_net = pd.Series(0, index=df_stock.index)
            if not df_chip.empty:
                df_chip['date'] = pd.to_datetime(df_chip['date'])
                f_buy = df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['buy'].sum() - df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['sell'].sum()
                t_buy = df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['buy'].sum() - df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['sell'].sum()
                foreign_net = df_stock['date'].map(f_buy).fillna(0) / 1000.0
                trust_net = df_stock['date'].map(t_buy).fillna(0) / 1000.0

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

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.write("**💰 估值指標 (PE / PB)**")
                if not df_per.empty and 'PER' in df_per.columns:
                    st.info(f"- 本益比 (PE): `{df_per['PER'].iloc[-1]} 倍`\n- 股價淨值比 (PB): `{df_per['PBR'].iloc[-1]} 倍`")
            with col_f2:
                st.write("**📊 近期月營收**")
                if not df_rev.empty and 'revenue' in df_rev.columns:
                    st.dataframe(df_rev.tail(5)[['revenue_year', 'revenue_month', 'revenue']], hide_index=True)

# Tab 3: 歷史紀錄對照
with tab3:
    st.subheader("📜 歷史預測紀錄與實戰對照明細")
    if not df_all.empty:
        st.dataframe(df_all.sort_values(by='id', ascending=False), use_container_width=True, hide_index=True)
