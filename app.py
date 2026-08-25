import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

# 設定 Streamlit 頁面標題與佈局
st.set_page_config(page_title="台股 AI 量化智庫儀表板", page_icon="📈", layout="wide")

# 自訂 CSS 提升視覺質感
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

st.title("📈 台股 AI 量化智庫與部位風控儀表板")
st.caption("集成機器學習、三層屏障法、大盤體制過濾 (Regime Filter) 與半凱利資金部位配置 (Half-Kelly)")

conn = sqlite3.connect("paper_trading.db")

# ==========================================
# 1. 頂部總體盲測戰績統計
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
# 2. AI 當前大盤體制與策略氣象燈
# ==========================================
st.subheader("🤖 AI 大盤體制與策略氣象診斷")

today_str = datetime.now().strftime('%Y-%m-%d')
df_today = pd.read_sql(f"SELECT * FROM predictions WHERE predict_date='{today_str}'", conn)

current_regime = 'NORMAL'
if not df_today.empty and 'market_regime' in df_today.columns:
    val = df_today['market_regime'].iloc[0]
    if pd.notna(val):
        current_regime = val

col_status1, col_status2 = st.columns([1, 2])

with col_status1:
    if current_regime == 'BEAR':
        st.error("🚨 **大盤體制：空頭/高風險體制**")
        st.write("**當前策略：強攻防守 (門檻 ≥ 64.0%)**")
    elif current_regime == 'BULL':
        st.success("🟢 **大盤體制：強勢多頭體制**")
        st.write("**當前策略：順勢攻擊 (門檻 ≥ 56.0%)**")
    else:
        st.info("🟡 **大盤體制：震盪整理體制**")
        st.write("**當前策略：常態穩健 (門檻 ≥ 58.0%)**")

with col_status2:
    st.markdown("**🔍 風險環境與部位管控評估**")
    if current_regime == 'BEAR':
        st.write("大盤加權指數跌破月線/季線，系統性風險偏高。AI 系統已啟動**強攻防守機制**，自動調高選股門檻並調降建議下單部位比例，防止逆勢接刀風險。")
    elif current_regime == 'BULL':
        st.write("大盤維持多頭排列且站穩生命線，市場環境極佳。系統已放寬門檻，並透過半凱利公式為高勝率標的計算最佳資金配置比例。")
    else:
        st.write("大盤呈區間震盪整理，個股表現分化。AI 系統採用標準門檻與嚴格 ATR 動態停利/停損屏障進行過濾。")

st.divider()

# ==========================================
# 3. 今日精選標的與凱利部位配置建議
# ==========================================
st.subheader(f"🔥 今日 ({today_str}) AI 精選標的與凱利部位配置建議")

if not df_today.empty:
    for _, row in df_today.iterrows():
        upside = round(((row['tp_price'] - row['latest_price']) / row['latest_price']) * 100, 1)
        downside = round(((row['latest_price'] - row['sl_price']) / row['latest_price']) * 100, 1)
        rr_ratio = round(upside / (downside + 1e-6), 2)
        
        # 營收 YoY 格式化防護處理
        rev_yoy_raw = row.get('revenue_yoy', None)
        if pd.notna(rev_yoy_raw) and rev_yoy_raw != 'N/A':
            rev_val = float(rev_yoy_raw)
            rev_str = f"+{rev_val:.1f}%" if rev_val > 0 else f"{rev_val:.1f}%"
        else:
            rev_str = "未提供"

        pe_val = row.get('pe_ratio', 'N/A')
        pos_size = row.get('position_size', 5.0)

        st.markdown(f"""
        <div class="stock-card">
            <h3 style="margin:0; color:#1e3a8a;">📌 {row['stock_name']} ({row['stock_id']})</h3>
            <p style="margin-top:5px; color:#475569;">
                <b>最新收盤價：</b> NT$ {row['latest_price']} ｜ 
                <b>AI 預估勝率：</b> <span style="color:#d97706; font-size:18px; font-weight:bold;">{row['ai_win_rate']}%</span> ｜
                <b>建議部位配置：</b> <span style="color:#2563eb; font-size:18px; font-weight:bold;">{pos_size}% 總資金</span> ｜
                <b>營收 YoY：</b> <span style="color:#16a34a; font-weight:bold;">{rev_str}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander(f"📖 點擊查看 {row['stock_name']} 的風控細節與算價報告"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**🎯 AI 交易價位與風險控制**")
                st.write(f"- **建議買入價**：`NT$ {row['buy_price']}`")
                st.write(f"- **ATR 目標停利價**：`NT$ {row['tp_price']}` (預期 +{upside}%)")
                st.write(f"- **ATR 防守停損價**：`NT$ {row['sl_price']}` (風險 -{downside}%)")
                st.write(f"- **預期風報比 (R/R Ratio)**：`{rr_ratio}`")
            
            with col_b:
                st.markdown("**💡 凱利部位資金管理算價**")
                st.write(f"- **計算模式**：半凱利公式 (Half-Kelly Criterion)")
                st.write(f"- **風控邏輯**：基於勝率 **{row['ai_win_rate']}%** 與風報比 **{rr_ratio}** 計算出之最適單筆資金上限，單檔上限不超過 15%。")
                st.write(f"- **風控建議**：當前大盤體制為 `{row.get('market_regime', 'NORMAL')}`，建議最大曝險不超過總資產的 **{pos_size}%**。")
            
            st.write("---")
else:
    st.info("今日市場經 AI 與大盤體制過濾後無符合條件標的，或今日盤後數據尚未更新。")

st.divider()

# ==========================================
# 4. 歷史紀錄全明細表 (美化排版)
# ==========================================
st.subheader("📜 歷史預測紀錄與實戰對照明細")

if not df_all.empty:
    df_display = df_all.sort_values(by='id', ascending=False).copy()
    
    rename_dict = {
        'predict_date': '預測日期',
        'stock_id': '股票代碼',
        'stock_name': '股票名稱',
        'latest_price': '最新收盤價',
        'ai_win_rate': 'AI勝率(%)',
        'position_size': '建議部位(%)',
        'buy_price': '建議買價',
        'tp_price': '停利價(ATR)',
        'sl_price': '停損價(ATR)',
        'revenue_yoy': '營收YoY(%)',
        'pe_ratio': '本益比(PE)',
        'market_regime': '大盤體制',
        'status': '驗證狀態'
    }
    
    display_cols = [c for c in rename_dict.keys() if c in df_display.columns]
    df_display = df_display[display_cols].rename(columns=rename_dict)
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "AI勝率(%)": st.column_config.NumberColumn(format="%.1f%%"),
            "建議部位(%)": st.column_config.NumberColumn(format="%.1f%%"),
            "最新收盤價": st.column_config.NumberColumn(format="NT$ %.2f"),
            "建議買價": st.column_config.NumberColumn(format="NT$ %.2f"),
            "停利價(ATR)": st.column_config.NumberColumn(format="NT$ %.2f"),
            "停損價(ATR)": st.column_config.NumberColumn(format="NT$ %.2f"),
        }
    )

conn.close()
