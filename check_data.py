import pandas as pd

train = pd.read_csv('train.csv')
sub = pd.read_csv('submission_advanced.csv')

print('='*80)
print('실제 데이터 vs 예측 데이터 비교')
print('='*80)

print('\n【실제 데이터 통계】')
print(train['value'].describe())
print(f'\n최대값: {train.value.max():,}')
print(f'평균값: {train.value.mean():,.0f}')

print('\n【예측 데이터 통계】')
print(sub['value'].describe())
print(f'\n최대값: {sub.value.max():,}')
print(f'평균값: {sub.value.mean():,.0f}')

print('\n【비율 비교】')
print(f'예측 평균 / 실제 평균 = {sub.value.mean() / train.value.mean():.1f}배')
print(f'예측 최대 / 실제 최대 = {sub.value.max() / train.value.max():.1f}배')

print('\n【품목별 실제 평균값 TOP 10】')
item_means = train.groupby('item_id')['value'].mean().sort_values(ascending=False).head(10)
for item, val in item_means.items():
    print(f'  {item}: {val:,.0f}')

print('\n【FCYBOAXC 품목 분석】')
fcyb = train[train['item_id'] == 'FCYBOAXC']['value']
if len(fcyb) > 0:
    print(f'  평균: {fcyb.mean():,.0f}')
    print(f'  최대: {fcyb.max():,}')
    print(f'  최소: {fcyb.min():,}')
    print(f'  최근 3개월: {fcyb.tail(3).values}')
else:
    print('  데이터 없음')

print('\n【예측에서 FCYBOAXC로 향하는 쌍】')
fcyb_preds = sub[sub['following_item_id'] == 'FCYBOAXC']
print(f'  개수: {len(fcyb_preds)}개')
print(f'  평균 예측값: {fcyb_preds.value.mean():,.0f}')
print(f'  최대 예측값: {fcyb_preds.value.max():,}')
