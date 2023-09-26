import pandas as pd
import matplotlib.pyplot as plt


########################################################################################################################################################################################
## 1. Load data
FS = pd.read_csv(f'dacon/심화 loaded data/FS_summary.csv')
FS = FS.set_index('Date').sort_index()
FS.index = pd.to_datetime(FS.index)
FS

## 1.1. 비주얼체크

columns_to_plot = [col for col in FS.columns if col != 'Date'] # 'Date' 컬럼을 제외한 나머지 컬럼들을 선택

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10)) # 2x3 형태로 그래프 그리기
for ax, column in zip(axes.ravel(), columns_to_plot):
    ax.plot(FS.index, FS[column], label=column)
    ax.set_title(column)
    ax.legend()
plt.tight_layout()
plt.show()


##
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# 2020-07-02부터 2020-09-08까지의 날짜 생성
date_extension = pd.date_range(start="2023-07-02", end="2023-09-08", freq='D')

# 기존 FS 데이터만을 사용하여 인덱스의 빈도 추론
inferred_freq = pd.infer_freq(FS.dropna().index)
FS.index.freq = inferred_freq

# FS 데이터에 새로운 날짜 범위 추가
FS = FS.reindex(FS.index.union(date_extension))

# 각 변수에 대해 지수평활법 적용
for column in FS.columns:
    model = SimpleExpSmoothing(FS[column]).fit(smoothing_level=0.2, optimized=False)
    FS[column] = model.fittedvalues

FS.tail(10)  # 결과 확인
