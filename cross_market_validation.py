"""
跨市場驗證腳本（研究/驗證用途，不是每日排程的一部分，也不會寫入 paper_trading.db）。

把 paper_trading.py 裡台股引擎用的 score_technical() / score_fundamental() 邏輯，原封不動套用在
美股與日股個股的歷史資料上，檢查「總分高的那天，之後幾天報酬是不是真的比較好」——用來驗證這套
評分邏輯是不是只是矇中台股的特性，還是真的有跨市場的普適性。

跟 paper_trading.py 的差異（刻意簡化，讓這支腳本能獨立在美日市場跑）：
- 不用籌碼面：美日沒有對應台灣三大法人格式的免費資料源
- 不用跨市場因子：這裡本身就是在測試美日市場，沒有意義再疊加美日領先指標
- 基本面用季頻營收 YoY 近似（yfinance 只有季報，沒有台股月營收那種月頻率資料），
  PE 百分位資料不足時會用 score_fundamental() 本身內建的中性 fallback（不會出錯，只是那個子項不計分）

用法：python cross_market_validation.py
結果會印出分數區間對應的「未來 N 個交易日上漲機率」與「平均報酬」，並存成
cross_market_validation_results.csv 供進一步分析。
"""
import pandas as pd
import numpy as np
import yfinance as yf
import time

from paper_trading import score_technical, score_fundamental

FORWARD_WINDOW = 10  # 看未來 10 個交易日的報酬
SCORE_BUCKETS = [(0, 40), (40, 55), (55, 70), (70, 85), (85, 101)]
TECH_WEIGHT = 0.6
FUND_WEIGHT = 0.4

US_UNIVERSE = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet', 'AMZN': 'Amazon',
    'NVDA': 'Nvidia', 'META': 'Meta Platforms', 'TSLA': 'Tesla', 'JPM': 'JPMorgan Chase',
    'JNJ': 'Johnson & Johnson', 'XOM': 'Exxon Mobil', 'V': 'Visa', 'PG': 'Procter & Gamble',
    'HD': 'Home Depot', 'MA': 'Mastercard', 'UNH': 'UnitedHealth', 'DIS': 'Disney',
    'BAC': 'Bank of America', 'ADBE': 'Adobe', 'CRM': 'Salesforce', 'NFLX': 'Netflix',
    'KO': 'Coca-Cola', 'PEP': 'PepsiCo', 'COST': 'Costco', 'INTC': 'Intel',
    'AMD': 'AMD', 'QCOM': 'Qualcomm', 'ORCL': 'Oracle', 'IBM': 'IBM',
    'GE': 'General Electric', 'CAT': 'Caterpillar',
}

JP_UNIVERSE = {
    '7203.T': 'Toyota', '6758.T': 'Sony', '9984.T': 'SoftBank Group',
    '6861.T': 'Keyence', '8306.T': 'MUFG', '9432.T': 'NTT', '6098.T': 'Recruit',
    '4063.T': 'Shin-Etsu Chemical', '6501.T': 'Hitachi', '6902.T': 'Denso',
    '7267.T': 'Honda', '8316.T': 'Sumitomo Mitsui FG', '4502.T': 'Takeda',
    '6367.T': 'Daikin', '9433.T': 'KDDI', '8035.T': 'Tokyo Electron',
    '6752.T': 'Panasonic', '7751.T': 'Canon', '4519.T': 'Chugai Pharmaceutical',
    '8058.T': 'Mitsubishi Corp', '2914.T': 'Japan Tobacco',
}

def fetch_price_history(ticker):
    try:
        df = yf.download(ticker, period='3y', progress=False, auto_adjust=False)
        time.sleep(0.2)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={'Date': 'date'})
        return df
    except Exception:
        return None

def fetch_quarterly_yoy(ticker):
    """用季報營收年增率當基本面的 YoY 近似值，抓不到就回傳 None（score_fundamental 會給中性分）。"""
    try:
        t = yf.Ticker(ticker)
        fin = t.quarterly_income_stmt
        time.sleep(0.2)
        if fin is None or fin.empty or 'Total Revenue' not in fin.index:
            return None
        rev = fin.loc['Total Revenue'].dropna().sort_index()
        if len(rev) < 5:
            return None
        latest, year_ago = rev.iloc[-1], rev.iloc[-5]
        if year_ago == 0 or pd.isna(year_ago) or pd.isna(latest):
            return None
        return float((latest / year_ago - 1) * 100)
    except Exception:
        return None

def evaluate_universe(universe, label):
    records = []
    empty_per = pd.Series(dtype=float)
    for ticker, name in universe.items():
        df = fetch_price_history(ticker)
        if df is None or len(df) < 260:
            print(f"  ⚠️ {label} {ticker}（{name}）歷史資料不足，跳過。")
            continue
        yoy = fetch_quarterly_yoy(ticker)

        stock_hits = 0
        for i in range(120, len(df) - FORWARD_WINDOW):
            window_df = df.iloc[:i + 1]
            tech_score, _ = score_technical(window_df, volatility_regime='NORMAL')
            if tech_score is None:
                continue
            fund_score, _ = score_fundamental(yoy, False, empty_per)
            total_score = round(tech_score * TECH_WEIGHT + fund_score * FUND_WEIGHT, 1)

            entry_price = float(df['Close'].iloc[i])
            future_price = float(df['Close'].iloc[i + FORWARD_WINDOW])
            fwd_return = (future_price / entry_price - 1) * 100

            records.append({
                'market': label, 'ticker': ticker, 'name': name,
                'date': df['date'].iloc[i], 'total_score': total_score,
                'fwd_return_10d': fwd_return,
            })
            stock_hits += 1
        print(f"  ✅ {label} {ticker}（{name}）：{stock_hits} 個有效樣本日。")
    return records

def print_bucket_report(df, label):
    print(f"\n=== {label}（樣本數 {len(df)}）===")
    for lo, hi in SCORE_BUCKETS:
        bucket = df[(df['total_score'] >= lo) & (df['total_score'] < hi)]
        if bucket.empty:
            continue
        win_rate = (bucket['fwd_return_10d'] > 0).mean() * 100
        avg_return = bucket['fwd_return_10d'].mean()
        print(f"  分數 {lo:>3}-{hi:<3}：樣本 {len(bucket):>5} 筆｜{FORWARD_WINDOW}日後上漲機率 {win_rate:5.1f}%｜平均報酬 {avg_return:+6.2f}%")
    if len(bucket := df.dropna(subset=['total_score', 'fwd_return_10d'])) > 1:
        corr = bucket['total_score'].corr(bucket['fwd_return_10d'])
        print(f"  總分與{FORWARD_WINDOW}日後報酬的相關係數：{corr:.3f}（越接近 1 代表分數越能預測未來報酬，0 代表無關）")

def main():
    print("開始跨市場驗證：套用台股引擎的技術面+基本面邏輯到美股/日股個股歷史資料...\n")

    print(f"📡 抓取美股 {len(US_UNIVERSE)} 檔歷史資料並評分...")
    us_records = evaluate_universe(US_UNIVERSE, 'US')

    print(f"\n📡 抓取日股 {len(JP_UNIVERSE)} 檔歷史資料並評分...")
    jp_records = evaluate_universe(JP_UNIVERSE, 'JP')

    all_records = us_records + jp_records
    if not all_records:
        print("\n沒有取得任何有效資料，無法驗證。")
        return

    df = pd.DataFrame(all_records)
    df.to_csv('cross_market_validation_results.csv', index=False)
    print(f"\n✅ 共 {len(df)} 筆樣本，明細已存成 cross_market_validation_results.csv")

    print_bucket_report(df[df['market'] == 'US'], 'US 市場')
    print_bucket_report(df[df['market'] == 'JP'], 'JP 市場')
    print_bucket_report(df, '美日合併')

if __name__ == "__main__":
    main()
