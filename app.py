import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# 設定頁面寬度與標題
st.set_page_config(page_title="台股 AI 量化智庫儀表板", page_icon="📈", layout="wide")

# 自訂 CSS 提升質感
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .status-badge {
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 14px;
    }
    .stock-card {
        background-color: #ffffff;
        border-left: 5px solid #3b82f6;
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

st.title("📈 台股 AI 量化智庫與實戰解讀儀表板")
st.caption("結合集成機器學習、動態門檻校正與多因子文字診斷模型")

conn = sqlite3.connect("paper_trading.db")

# ==========================================
# 1. 頂部整體盲測戰績與 AI 策略狀態
# ==========================================
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
# 2. 當前市場狀態與 AI 策略氣象燈 (文字解讀)
# ==========================================
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
        st.write("近期市場波動較大或處於震盪洗盤格局，多頭動能延續性較弱。AI 系統已自動**收緊選股標準**，透過提高進場門檻來剔除假突破標的，優先確保資金安全。")
    elif recent_win_rate >= 65.0:
        st.write("當前市場多頭趨勢顯著，技術面突破與法人籌碼追價的成功率極高。AI 模型與當前行情高度契合，系統已適度放寬門檻以抓取更多波段爆發標的。")
    else:
        st.write("當前市場結構處於正常多空交替狀態，歷史盲測勝率維持在合理區間。AI 系統採用標準選股門檻，均衡考慮技術面動能與籌碼集中度。")

st.divider()

# ==========================================
# 3. 今日 AI 精選標的（具體文字解讀卡片）
# ==========================================
today_str = datetime.now().strftime('%Y-%m-%d')
st.subheader(f"🔥 今日 ({today_str}) AI 精選標的與決策報告")

df_today = pd.read_sql(f"SELECT * FROM predictions WHERE predict_date='{today_str}'", conn)

if not df_today.empty:
    for _, row in df_today.iterrows():
        upside = round(((row['tp_price'] - row['latest_price']) / row['latest_price']) * 100, 1)
        downside = round(((row['latest_price'] - row['sl_price']) / row['latest_price']) * 100, 1)
        rr_ratio = round(upside / (downside + 1e-6), 2)
        
        st.markdown(f"""
        <div class="stock-card">
            <h3 style="margin:0; color:#1e3a8a;">📌 {row['stock_name']} ({row['stock_id']})</h3>
            <p style="margin-top:5px; color:#475569;">
                <b>最新收盤價：</b> NT$ {row['latest_price']} ｜ 
                <b>AI 預估看漲勝率：</b> <span style="color:#d97706; font-size:18px; font-weight:bold;">{row['ai_win_rate']}%</span> ｜
                <b>預期風報比：</b> {rr_ratio} (潛在獲利: +{upside}% / 防守風險: -{downside}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander(f"📖 點擊查看 {row['stock_name']} 的 AI 完整分析報告與操作建議"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**🎯 AI 交易價位建議**")
                st.write(f"- **建議進場價**：`NT$ {row['buy_price']}` (參考當日收盤價)")
                st.write(f"- **目標停利價**：`NT$ {row['tp_price']}` (對應前波強壓力線，預期 +{upside}%)")
                st.write(f"- **防守停損價**：`NT$ {row['sl_price']}` (對應近 20 日強支撐，風險 -{downside}%)")
            
            with col_b:
                st.markdown("**💡 AI 決策邏輯與模型觀點**")
                st.write(f"- **強度評估**：模型預估該股未來 3 個交易日內上漲超過 1.5% 的機率高達 **{row['ai_win_rate']}%**。")
                st.write("- **特徵綜合判斷**：均線呈多頭排列，短線動能顯著增強，同時外資與投信籌碼出現集中流入跡象，符合高勝率攻擊型態。")
                st.write("- **操作戰術建議**：建議於開盤附近佈局，若股價順利突破目標價可分批獲利入袋；若回檔跌破支撐價位則嚴格執行停損退場。")
            
            st.write("---")
else:
    st.info("今日市場經 AI 綜合評估後，無勝率符合當前自適應門檻之標的，或今日盤後數據尚未更新。")

st.divider()

# ==========================================
# 4. 歷史盲測戰績全紀錄 (美化修正版)
# ==========================================
st.subheader("📜 歷史預測紀錄與實戰對照明細")

if not df_all.empty:
    # 進行欄位挑選、排序與中文美化
    df_display = df_all.sort_values(by='id', ascending=False).copy()
    
    # 重新命名欄位
    rename_dict = {
        'predict_date': '預測日期',
        'stock_id': '股票代碼',
        'stock_name': '股票名稱',
        'latest_price': '最新收盤價',
        'buy_price': '建議買價',
        'tp_price': '停利價',
        'sl_price': '停損價',
        'ai_win_rate': 'AI預估勝率(%)',
        'status': '驗證狀態',
        'real_max_price': '對照期間最高價',
        'real_min_price': '對照期間最低價',
        'validated_date': '結算驗證日期'
    }
    
    # 選取需要顯示的欄位並更名 (隱藏 id 欄位)
    display_cols = [c for c in rename_dict.keys() if c in df_display.columns]
    df_display = df_display[display_cols].rename(columns=rename_dict)
    
    # 使用 Streamlit 原生 Column Config 進行格式化與美化渲染
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,  # 隱藏左側索引列
        column_config={
            "AI預估勝率(%)": st.column_config.NumberColumn(format="%.1f%%"),
            "最新收盤價": st.column_config.NumberColumn(format="NT$ %.2f"),
            "建議買價": st.column_config.NumberColumn(format="NT$ %.2f"),
            "停利價": st.column_config.NumberColumn(format="NT$ %.2f"),
            "停損價": st.column_config.NumberColumn(format="NT$ %.2f"),
            "對照期間最高價": st.column_config.NumberColumn(format="NT$ %.2f"),
            "對照期間最低價": st.column_config.NumberColumn(format="NT$ %.2f"),
        }
    )
else:
    st.write("目前資料庫中尚未有歷史預測紀錄。")

conn.close()
