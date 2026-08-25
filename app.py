import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="台股 AI 量化智庫儀表板", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stock-card {
        background-color: #ffffff;
        border-left: 5px solid #2563eb;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 台股 AI 量化智庫與基本面解讀儀表板")
st.caption("結合集成機器學習、基本面雙層過濾（營收YoY/本益比）與自適應門檻校正機制")

conn = sqlite3.connect("paper_trading.db")

# 1. 頂部整體盲測戰績
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

# 2. AI 策略與氣象燈
st.subheader("🤖 AI 當前策略與市場氣象診斷")

df_recent = pd.read_sql("SELECT status FROM predictions WHERE status!='PENDING' ORDER BY id DESC LIMIT 20", conn)
recent_total = len(df_recent)
recent_wins = len(df_recent[df_recent['status'] == 'WIN (成功停利)'])
recent_win_rate = (recent_wins / recent_total * 100) if recent_total > 0 else 50.0

col_status1, col_status2 = st.columns([1, 2])

with col_status1:
    if recent_win_rate < 45.0:
        st.error("🛡️ **當前策略模式：強攻防守模式**")
        st.write("**當前進場勝率門檻：`63.0%`**")
    elif recent_win_rate >= 65.0:
        st.success("⚔️ **當前策略模式：順勢攻擊模式**")
        st.write("**當前進場勝率門檻：`56.0%`**")
    else:
        st.info("⚖️ **當前策略模式：常態穩健模式**")
        st.write("**當前進場勝率門檻：`58.0%`**")

with col_status2:
    st.markdown("**🔍 市場環境與 AI 診斷評估**")
    if recent_win_rate < 45.0:
        st.write("近期市場波動較大，多頭動能延續性較弱。AI 系統已自動**調高門檻並結合基本面硬性濾網**，嚴格過濾無業績支撐的個股以降低風險。")
    elif recent_win_rate >= 65.0:
        st.write("當前市場多頭趨勢顯著，籌碼與業績雙優標的表現亮眼。AI 系統已適度放寬門檻，積極捕捉波段強勢個股。")
    else:
        st.write("當前市場結構處於正常多空交替狀態。AI 系統採用標準門檻，均衡考慮技術面動量、法人籌碼與基本面營收表現。")

st.divider()

# 3. 今日 AI 精選標的 (基本面解讀卡片)
today_str = datetime.now().strftime('%Y-%m-%d')
st.subheader(f"🔥 今日 ({today_str}) AI 精選標的與基本面決策報告")

df_today = pd.read_sql(f"SELECT * FROM predictions WHERE predict_date='{today_str}'", conn)

if not df_today.empty:
    for _, row in df_today.iterrows():
        upside = round(((row['tp_price'] - row['latest_price']) / row['latest_price']) * 100, 1)
        downside = round(((row['latest_price'] - row['sl_price']) / row['latest_price']) * 100, 1)
        rr_ratio = round(upside / (downside + 1e-6), 2)
        
        rev_yoy = row.get('revenue_yoy', 'N/A')
        pe_val = row.get('pe_ratio', 'N/A')

        st.markdown(f"""
        <div class="stock-card">
            <h3 style="margin:0; color:#1e3a8a;">📌 {row['stock_name']} ({row['stock_id']})</h3>
            <p style="margin-top:5px; color:#475569;">
                <b>最新收盤價：</b> NT$ {row['latest_price']} ｜ 
                <b>AI 預估勝率：</b> <span style="color:#d97706; font-size:18px; font-weight:bold;">{row['ai_win_rate']}%</span> ｜
                <b>營收 YoY：</b> <span style="color:#16a34a; font-weight:bold;">+{rev_yoy}%</span> ｜
                <b>本益比 (PE)：</b> {pe_val} 倍
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander(f"📖 點擊查看 {row['stock_name']} 的 AI 多維度完整分析報告"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**🎯 AI 交易價位建議**")
                st.write(f"- **建議進場價**：`NT$ {row['buy_price']}` (參考當日收盤價)")
                st.write(f"- **目標停利價**：`NT$ {row['tp_price']}` (預期 +{upside}%)")
                st.write(f"- **防守停損價**：`NT$ {row['sl_price']}` (風險 -{downside}%)")
            
            with col_b:
                st.markdown("**💡 AI 基本面與籌碼邏輯**")
                st.write(f"- **業績成長**：最新營收年增率為 **+{rev_yoy}%**，展現明確本業基本面支撐。")
                st.write(f"- **估值安全度**：當前本益比為 **{pe_val} 倍**，位於合理估值區間。")
                st.write(f"- **綜合評價**：模型評估該股兼具「營收成長、技術突破與籌碼集中」三大特性，看漲勝率達 **{row['ai_win_rate']}%**。")
            
            st.write("---")
else:
    st.info("今日市場經 AI 與基本面雙重過濾後，無符合條件標的，或今日盤後數據尚未更新。")

st.divider()

# 4. 歷史明細紀錄
st.subheader("📜 歷史預測紀錄與實戰對照明細")

if not df_all.empty:
    df_display = df_all.sort_values(by='id', ascending=False).copy()
    
    rename_dict = {
        'predict_date': '預測日期',
        'stock_id': '股票代碼',
        'stock_name': '股票名稱',
        'latest_price': '最新收盤價',
        'revenue_yoy': '營收YoY(%)',
        'pe_ratio': '本益比(PE)',
        'buy_price': '建議買價',
        'tp_price': '停利價',
        'sl_price': '停損價',
        'ai_win_rate': 'AI勝率(%)',
        'status': '驗證狀態',
        'validated_date': '結算日期'
    }
    
    display_cols = [c for c in rename_dict.keys() if c in df_display.columns]
    df_display = df_display[display_cols].rename(columns=rename_dict)
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "AI勝率(%)": st.column_config.NumberColumn(format="%.1f%%"),
            "最新收盤價": st.column_config.NumberColumn(format="NT$ %.2f"),
            "建議買價": st.column_config.NumberColumn(format="NT$ %.2f"),
            "停利價": st.column_config.NumberColumn(format="NT$ %.2f"),
            "停損價": st.column_config.NumberColumn(format="NT$ %.2f"),
        }
    )

conn.close()
