import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as pgo
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# 設定 Streamlit 頁面標題與佈局
st.set_page_config(page_title="台股 AI 量化智庫與互動終端", page_icon="📈", layout="wide")

DB_NAME = "paper_trading.db"
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoiMDUxOGNoaXl1QGdtYWlsLmNvbSIsImVtYWlsIjoiMDUxOGNoaXl1QGdtYWlsLmNvbSIsInRva2VuX3ZlcnNpb24iOjAsImV4cCI6MTc4ODI0MDUwOH0.dNGO-ZUPpWW30mfiUdwMqIJV-v2bqShtiLJsoy4vh7I"

@st.cache_resource
def get_finmind_loader():
    dl = DataLoader()
    try:
        dl.login_by_token(token=FINMIND_TOKEN)
    except Exception:
        pass
    return dl

dl = get_finmind_loader()

# ==========================================
# 0. 自選股 (Watchlist) 資料庫初始化與操作函數
# ==========================================
def init_watchlist_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            stock_id TEXT PRIMARY KEY,
            stock_name TEXT,
            added_date TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_watchlist_db()

def add_to_watchlist(stock_id):
    stock_id = stock_id.strip()
    if not stock_id:
        return False, "請輸入股票代碼！"
    
    # 透過 FinMind 驗證代碼與名稱
    try:
        info = dl.taiwan_stock_info()
        matched = info[info['stock_id'] == stock_id]
        if matched.empty:
            return False, f"查無股票代碼 {stock_id}"
        stock_name = matched['stock_name'].iloc[0]
    except Exception:
        stock_name = f"股票 {stock_id}"

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    today_str = datetime.now().strftime('%Y-%m-%d')
    try:
        cursor.execute("INSERT INTO watchlist (stock_id, stock_name, added_date) VALUES (?, ?, ?)",
                       (stock_id, stock_name, today_str))
        conn.commit()
        conn.close()
        return True, f"成功加入自選股：{stock_name} ({stock_id})"
    except sqlite3.IntegrityError:
        conn.close()
        return False, f"股票 {stock_id} 已在關注清單中！"

def remove_from_watchlist(stock_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM watchlist WHERE stock_id=?", (stock_id,))
    conn.commit()
    conn.close()

def get_watchlist():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM watchlist", conn)
    conn.close()
    return df

# ==========================================
# 1. 側邊欄：自選股管理 (Watchlist Control)
# ==========================================
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

# ==========================================
# 2. 主頁面標題與總覽數據
# ==========================================
st.title("📈 台股 AI 量化智庫與互動視覺化終端")
st.caption("結合 AI 選股模型、個人自選股清單、動態 K 線圖與法人籌碼/財報圖解")

conn = sqlite3.connect(DB_NAME)
df_all = pd.read_sql("SELECT * FROM predictions", conn)
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

# ==========================================
# 3. 頁籤分類：[AI 今日選股] 與 [個人關注個股圖表診斷]
# ==========================================
tab1, tab2, tab3 = st.tabs(["🔥 今日 AI 精選決策", "🔍 個股 K 線 / 籌碼 / 財報深度圖解", "📜 歷史預測對照表"])

today_str = datetime.now().strftime('%Y-%m-%d')
df_today = pd.read_sql(f"SELECT * FROM predictions WHERE predict_date='{today_str}'", conn)

# ---------------- Tab 1: AI 精選 ----------------
with tab1:
    st.subheader(f"🤖 今日 ({today_str}) AI 精選標的與建議")
    if not df_today.empty:
        for _, row in df_today.iterrows():
            upside = round(((row['tp_price'] - row['latest_price']) / row['latest_price']) * 100, 1)
            downside = round(((row['latest_price'] - row['sl_price']) / row['latest_price']) * 100, 1)
            rr_ratio = round(upside / (downside + 1e-6), 2)
            
            rev_yoy_raw = row.get('revenue_yoy', None)
            if pd.notna(rev_yoy_raw) and rev_yoy_raw != 'N/A':
                rev_val = float(rev_yoy_raw)
                rev_str = f"+{rev_val:.1f}%" if rev_val > 0 else f"{rev_val:.1f}%"
            else:
                rev_str = "未提供"

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
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"- **建議買入價**：`NT$ {row['buy_price']}`")
                    st.write(f"- **ATR 目標停利價**：`NT$ {row['tp_price']}` (預期 +{upside}%)")
                    st.write(f"- **ATR 防守停損價**：`NT$ {row['sl_price']}` (風險 -{downside}%)")
                with col_b:
                    st.write(f"- **風報比 (R/R Ratio)**：`{rr_ratio}`")
                    st.write(f"- **本益比 (PE)**：`{row.get('pe_ratio', 'N/A')} 倍`")
                    st.write(f"- **大盤當前體制**：`{row.get('market_regime', 'NORMAL')}`")
    else:
        st.info("今日市場經 AI 與風控過濾後無符合進場條件之標的，建議觀望保持現金。")

# ---------------- Tab 2: 互動 K 線 / 籌碼 / 財報圖解 ----------------
with tab2:
    st.subheader("📊 互動式技術面、籌碼面與基本面圖解")
    
    # 建立可選股票選單 (結合 AI 今日選股 + 個人關注清單)
    stock_options = {}
    if not df_today.empty:
        for _, r in df_today.iterrows():
            stock_options[r['stock_id']] = f"[AI 推薦] {r['stock_name']} ({r['stock_id']})"
    if not df_wl.empty:
        for _, r in df_wl.iterrows():
            if r['stock_id'] not in stock_options:
                stock_options[r['stock_id']] = f"[關注清單] {r['stock_name']} ({r['stock_id']})"
    
    # 預設選單防呆
    if not stock_options:
        stock_options['2330'] = "[熱門預設] 台積電 (2330)"

    selected_stock_id = st.selectbox("請選擇欲分析診斷的股票：", options=list(stock_options.keys()), format_func=lambda x: stock_options[x])

    if selected_stock_id:
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        with st.spinner(f"正在載入 {selected_stock_id} 的 K 線、籌碼與財報資料..."):
            # 1. 抓取日 K 線數據
            df_stock = dl.taiwan_stock_daily(stock_id=selected_stock_id, start_date=start_date)
            # 2. 抓取法人籌碼
            df_chip = dl.taiwan_stock_institutional_investors(stock_id=selected_stock_id, start_date=start_date)
            # 3. 抓取月營收
            df_rev = dl.taiwan_stock_month_revenue(stock_id=selected_stock_id, start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
            # 4. 抓取本益比/股淨比
            df_per = dl.taiwan_stock_per_pbr(stock_id=selected_stock_id, start_date=start_date)

        if not df_stock.empty:
            df_stock = df_stock.rename(columns={'max': 'High', 'min': 'Low', 'close': 'Close', 'open': 'Open', 'Trading_Volume': 'Volume'})
            df_stock['date'] = pd.to_datetime(df_stock['date'])
            df_stock = df_stock.sort_values('date').reset_index(drop=True)

            # 計算均線
            df_stock['MA5'] = df_stock['Close'].rolling(5).mean()
            df_stock['MA20'] = df_stock['Close'].rolling(20).mean()

            # 處理法人買賣超 (張數化 = 股數 / 1000)
            foreign_net, trust_net = pd.Series(0, index=df_stock.index), pd.Series(0, index=df_stock.index)
            if not df_chip.empty:
                df_chip['date'] = pd.to_datetime(df_chip['date'])
                f_buy = df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['buy'].sum() - df_chip[df_chip['name'] == 'Foreign_Investor'].groupby('date')['sell'].sum()
                t_buy = df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['buy'].sum() - df_chip[df_chip['name'] == 'Investment_Trust'].groupby('date')['sell'].sum()
                foreign_net = df_stock['date'].map(f_buy).fillna(0) / 1000.0
                trust_net = df_stock['date'].map(t_buy).fillna(0) / 1000.0

            # 繪製 Plotly 多圖表疊加 (K線 + 法人籌碼柱狀圖)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35],
                                subplot_titles=(f"{selected_stock_id} K 線與移動平均線 (MA5 / MA20)", "三大法人買賣超張數 (外資/投信)"))

            # 子圖 1: 蠟燭 K 線
            fig.add_trace(pgo.Candlestick(
                x=df_stock['date'], open=df_stock['Open'], high=df_stock['High'], low=df_stock['Low'], close=df_stock['Close'],
                name="日 K 線", increasing_line_color='#ef4444', decreasing_line_color='#22c55e'
            ), row=1, col=1)

            # 子圖 1: MA5 & MA20
            fig.add_trace(pgo.Scatter(x=df_stock['date'], y=df_stock['MA5'], mode='lines', name='5日均線 (MA5)', line=dict(color='#f59e0b', width=1.5)), row=1, col=1)
            fig.add_trace(pgo.Scatter(x=df_stock['date'], y=df_stock['MA20'], mode='lines', name='20日均線 (MA20)', line=dict(color='#3b82f6', width=1.5)), row=1, col=1)

            # 子圖 2: 法人籌碼柱狀圖
            fig.add_trace(pgo.Bar(x=df_stock['date'], y=foreign_net, name='外資買賣超(張)', marker_color='#8b5cf6'), row=2, col=1)
            fig.add_trace(pgo.Bar(x=df_stock['date'], y=trust_net, name='投信買賣超(張)', marker_color='#ec4899'), row=2, col=1)

            fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # ---------------- 財報基本面精華整理 ----------------
            st.markdown("#### 📑 財報與估值指標整理")
            col_f1, col_f2 = st.columns(2)

            with col_f1:
                st.write("**💰 估值指標 (PE / PB)**")
                if not df_per.empty and 'PER' in df_per.columns and 'PBR' in df_per.columns:
                    latest_per = df_per['PER'].iloc[-1]
                    latest_pbr = df_per['PBR'].iloc[-1]
                    st.info(f"- **最新本益比 (PE)**：`{latest_per} 倍` \n- **最新股價淨值比 (PB)**：`{latest_pbr} 倍`")
                else:
                    st.write("暫無最新估值數據。")

            with col_f2:
                st.write("**📊 近期月營收走勢 (千元)**")
                if not df_rev.empty and 'revenue' in df_rev.columns:
                    df_rev_display = df_rev.tail(5)[['revenue_year', 'revenue_month', 'revenue', 'revenue_year_growth_ratio']].copy()
                    df_rev_display.columns = ['年度', '月份', '營收(千元)', '年增率 YoY(%)']
                    st.dataframe(df_rev_display, hide_index=True, use_container_width=True)
                else:
                    st.write("暫無最新營收數據。")

# ---------------- Tab 3: 歷史對照表 ----------------
with tab3:
    st.subheader("📜 歷史預測紀錄與實戰對照明細")
    if not df_all.empty:
        df_display = df_all.sort_values(by='id', ascending=False).copy()
        rename_dict = {
            'predict_date': '預測日期', 'stock_id': '股票代碼', 'stock_name': '股票名稱',
            'latest_price': '最新收盤價', 'ai_win_rate': 'AI勝率(%)', 'position_size': '建議部位(%)',
            'buy_price': '建議買價', 'tp_price': '停利價(ATR)', 'sl_price': '停損價(ATR)',
            'revenue_yoy': '營收YoY(%)', 'pe_ratio': '本益比(PE)', 'market_regime': '大盤體制', 'status': '驗證狀態'
        }
        display_cols = [c for c in rename_dict.keys() if c in df_display.columns]
        df_display = df_display[display_cols].rename(columns=rename_dict)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

conn.close()
