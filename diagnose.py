import pandas as pd
import numpy as np

# 데이터 로드
sub_corrected = pd.read_csv('submission_corrected.csv')
granger = pd.read_csv('granger_results.csv')

print('='*80)
print('문제 진단 및 개선안')
print('='*80)

print(f'\n현재 제출 파일: {len(sub_corrected)}개 쌍')
print(f'Granger 전체: {len(granger)}개 쌍')

# 문제 1: 3000개 미달
print('\n【문제 1】 쌍 개수 부족')
print(f'  현재: {len(sub_corrected)}개')
print(f'  필요: 3,000개')
print(f'  부족: {3000 - len(sub_corrected)}개')

# 해결책: p-value 기준 완화
print('\n【해결책 1】 p-value 기준 완화')
for threshold in [0.10, 0.15, 0.20, 0.25, 0.30]:
    count = (granger['p_value'] < threshold).sum()
    print(f'  p < {threshold}: {count:,}개 쌍')

# 문제 2: F1 Score 낮음 (쌍이 안 맞음)
print('\n【문제 2】 F1 Score 낮음')
print('  가능한 원인:')
print('  1) Granger Causality가 실제 공행성과 다름')
print('  2) Lag 설정이 잘못됨')
print('  3) 너무 강한 인과관계만 선택 (p < 0.10)')

# 해결책: 상관계수 추가
print('\n【해결책 2】 상관계수 기반 쌍 추가')
print('  - Granger에서 누락된 쌍을 correlation으로 보완')
print('  - p-value 기준 완화 (0.10 → 0.20)')

# 문제 3: 예측값 과대
print('\n【문제 3】 예측값 과대')
print(f'  현재 평균: {sub_corrected["value"].mean():,.0f}')
print(f'  실제 평균: 1,739,442')
print(f'  비율: {sub_corrected["value"].mean() / 1739442:.2f}배')

print('\n【해결책 3】 예측값 재보정')
print('  - 전체 평균을 실제 데이터 평균으로 스케일링')
print('  - 또는 모델 재학습 (과적합 방지)')

# 추천 전략
print('\n' + '='*80)
print('📌 추천 개선 전략')
print('='*80)
print('1. p-value < 0.20으로 완화 → 약 4,000개 쌍')
print('2. 상위 3,000개 선택 (causality_score 기준)')
print('3. 예측값을 실제 평균에 맞게 스케일링')
print('4. 또는 상관계수 기반 쌍도 혼합 (Granger 70% + Correlation 30%)')
