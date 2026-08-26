"""
ml_trend_validation.py

Walk-forward 回測：驗證 ml_trend_model.py 的短期(4日)/中期(15日)機率預測，是不是真的對未來報酬
有預測力，而不是只是看起來訓練得很漂亮。不是每日排程的一部分，獨立執行、不寫入 paper_trading.db。

Walk-forward 的意思：把每支股票的歷史資料依時間切成 N 段，每一段只用「這一段之前」的資料訓練，
預測「這一段」，永遠不會用到未來資料訓練模型去預測過去——這樣算出來的命中率/相關係數才是
真正可信的樣本外表現，不是回頭看訓練集自己表現多好的假象。

用法：python ml_trend_validation.py
輸出：分機率區間的實際命中率/平均報酬對照表（跟 cross_market_validation.py 同樣風格），
以及明細 CSV（ml_trend_validation_results.csv）。
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

import ml_trend_model as mtm

N_FOLDS = 4
PROB_BUCKETS = [(0.0, 0.3), (0.3, 0.45), (0.45, 0.55), (0.55, 0.7), (0.7, 1.01)]

VALIDATION_UNIVERSE = {
    '2330': '台積電', '2317': '鴻海', '2454': '聯發科', '2308': '台達電',
    '2382': '廣達', '2303': '聯電', '2881': '富邦金', '2882': '國泰金',
    '5871': '中租-KY', '2603': '長榮', '2609': '陽明', '2412': '中華電',
    '2002': '中鋼', '1301': '台塑', '2357': '華碩', '2884': '玉山金',
    '3034': '聯詠', '3037': '欣興', '2892': '第一金', '1216': '統一',
    '2912': '統一超', '2408': '南亞科', '1101': '台泥', '2377': '微星',
    '2618': '長榮航', '3711': '日月光投控', '2379': '瑞昱', '2887': '台新新光金',
    '2801': '彰銀', '5880': '合庫金', '2886': '兆豐金', '9910': '豐泰',
}

PRICE_START_DATE = '2024-01-01'  # 約 2 年半歷史

def compute_forward_return(df, window):
    future_close = df['Close'].shift(-window)
    return (future_close / df['Close'] - 1) * 100

def walk_forward_evaluate(df, horizon, n_folds=N_FOLDS):
    label_col = 'label_short' if horizon == 'short' else 'label_medium'
    window = mtm.SHORT_WINDOW if horizon == 'short' else mtm.MEDIUM_WINDOW
    fwd_col = f'fwd_return_{horizon}'
    df[fwd_col] = compute_forward_return(df, window)

    valid = df.dropna(subset=mtm.FEATURE_COLUMNS + [label_col, fwd_col]).reset_index(drop=True)
    if len(valid) < 150:
        return []

    fold_size = len(valid) // n_folds
    records = []
    for fold in range(1, n_folds):
        train_end = fold * fold_size
        test_end = min(train_end + fold_size, len(valid))
        train_df = valid.iloc[:train_end]
        test_df = valid.iloc[train_end:test_end]
        if len(train_df) < 60 or test_df.empty or train_df[label_col].nunique() < 2:
            continue

        model = LGBMClassifier(n_estimators=60, learning_rate=0.03, max_depth=4, random_state=42, verbose=-1)
        model.fit(train_df[mtm.FEATURE_COLUMNS], train_df[label_col])
        probs = model.predict_proba(test_df[mtm.FEATURE_COLUMNS])[:, 1]

        for prob, (_, row) in zip(probs, test_df.iterrows()):
            records.append({
                'date': row['date'], 'win_prob': float(prob),
                'fwd_return': row[fwd_col], 'actual_label': row[label_col],
            })
    return records

def print_bucket_report(df, label):
    print(f"\n=== {label}（樣本數 {len(df)}）===")
    if df.empty:
        print("  無樣本。")
        return
    for lo, hi in PROB_BUCKETS:
        bucket = df[(df['win_prob'] >= lo) & (df['win_prob'] < hi)]
        if bucket.empty:
            continue
        win_rate = (bucket['fwd_return'] > 0).mean() * 100
        avg_return = bucket['fwd_return'].mean()
        hit_rate = bucket['actual_label'].mean() * 100
        print(f"  機率 {lo:.2f}-{hi:.2f}：樣本 {len(bucket):>5} 筆｜實際上漲機率 {win_rate:5.1f}%｜"
              f"平均報酬 {avg_return:+6.2f}%｜三重屏障命中率 {hit_rate:5.1f}%")
    corr = df['win_prob'].corr(df['fwd_return'])
    print(f"  預測機率與實際報酬的相關係數：{corr:.3f}（越接近 1 代表模型越準，0 代表無關）")

def main():
    print("開始 walk-forward 回測：驗證 ml_trend_model 的短期/中期預測是否真的有效...\n")

    print("📡 抓取全市場三大法人歷史資料（TWSE T86，逐日回溯）...")
    inst_data = mtm.fetch_historical_institutional_data(trading_days_back=500)
    print(f"✅ 取得 {len(inst_data)} 個交易日的法人資料。\n")

    print("📡 抓取全市場月營收歷史資料（MOPS，逐月回溯）...")
    revenue_data = mtm.fetch_historical_revenue_data(months_back=24)
    print(f"✅ 取得 {len(revenue_data)} 個月的營收資料。\n")

    print("📡 抓取大盤環境特徵（0050 代理）...")
    market_ctx = mtm.fetch_market_context_features()
    print(f"✅ 取得 {len(market_ctx)} 筆大盤資料。\n")

    short_records, medium_records = [], []
    print(f"📡 逐股建立特徵、跑 walk-forward 回測（共 {len(VALIDATION_UNIVERSE)} 檔）...")
    for stock_id, name in VALIDATION_UNIVERSE.items():
        df = mtm.build_training_dataframe(stock_id, PRICE_START_DATE, inst_data, revenue_data, market_ctx)
        if df is None:
            print(f"  ⚠️ {stock_id}（{name}）資料不足，跳過。")
            continue

        s_recs = walk_forward_evaluate(df.copy(), 'short')
        m_recs = walk_forward_evaluate(df.copy(), 'medium')
        for r in s_recs:
            r.update({'stock_id': stock_id, 'name': name})
        for r in m_recs:
            r.update({'stock_id': stock_id, 'name': name})
        short_records += s_recs
        medium_records += m_recs
        print(f"  ✅ {stock_id}（{name}）：短期樣本 {len(s_recs)}，中期樣本 {len(m_recs)}。")

    if not short_records and not medium_records:
        print("\n沒有取得任何有效樣本，無法驗證。")
        return

    df_short = pd.DataFrame(short_records)
    df_medium = pd.DataFrame(medium_records)
    if not df_short.empty:
        df_short['horizon'] = 'short'
    if not df_medium.empty:
        df_medium['horizon'] = 'medium'
    pd.concat([df_short, df_medium], ignore_index=True).to_csv('ml_trend_validation_results.csv', index=False)
    print(f"\n✅ 明細已存成 ml_trend_validation_results.csv")

    print_bucket_report(df_short, f"短期模型（未來 {mtm.SHORT_WINDOW} 個交易日）")
    print_bucket_report(df_medium, f"中期模型（未來 {mtm.MEDIUM_WINDOW} 個交易日）")

if __name__ == "__main__":
    main()
