import pandas as pd

print("제출 파일들 확인:\n")

files = [
    'submission_v3_final.csv',
    'submission_ultra.csv',
    'submission_advanced.csv',
    'submission_improved_v2.csv'
]

for fname in files:
    try:
        df = pd.read_csv(f'./{fname}')
        print(f"【{fname}】")
        print(f"  쌍 개수: {len(df):,}")
        print(f"  평균: {df['value'].mean():,.0f}")
        print(f"  최소: {df['value'].min():,}")
        print(f"  최대: {df['value'].max():,}")
        print(f"  음수: {(df['value'] < 0).sum()}개")
        print()
    except:
        print(f"【{fname}】 - 파일 없음\n")
