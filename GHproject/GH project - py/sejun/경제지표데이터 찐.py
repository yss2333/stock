import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import seaborn as sns
import yfinance as yf


sns.set_style('whitegrid')

## 주식 데이터 불러오기
df = fdr.DataReader('AAPL', '2020-01-02', '2023-09-08')
print(df)

adj_close_df = df[['Adj Close']]
print(adj_close_df)
start = '2020-01-02'
end = '2023-09-08'

## 미국 달러 환율
dxy = fdr.DataReader('DX-Y.NYB', '2020-01-02','2023-09-08')
dxy_series = dxy['Adj Close']

# Convert the Series to a DataFrame
dxy_df = dxy_series.to_frame(name='DXY.Adj Close')
dxy = dxy[['Adj Close']]
dxy = dxy.rename(columns={'Adj Close': 'DXY.Adj Close'})
dxy
print(dxy)

from fredapi import Fred
fred = Fred(api_key = '4c55d0ee6170369793707da4cba1b7be')

## 미국 국채 금리 (20년, 10년, 5년, 1년)
dgs2 = fred.get_series('DGS2', observation_start=start, observation_end=end)
dgs5 = fred.get_series('DGS5', observation_start=start, observation_end=end)
dgs10 = fred.get_series('DGS10', observation_start=start, observation_end=end)


DGS = pd.concat([dgs2, dgs5,dgs10], axis=1)
DGS.columns = ['2-year', '5-year', '10-year']
DGS.index.name = 'Date'
print(DGS)
graph = DGS.plot(title="US Treasury Bond Rates (from )")
graph.set_ylabel("Interest Rate (%)")
graph.axhline(1.5, ls='--', color='r')

# 미국 장단기 금리차 | 금리차가 0에 가까워지거나 음수가 되면 (인버전), 이는 종종 경제의 둔화 또는 경기침체를 앞두고 있다는 시장의 예상을 반영하는 것
T10Y2Y = fdr.DataReader('FRED:T10Y2Y', start, end)
T10Y2Y.index.name = 'Date'
print(T10Y2Y)
graph=T10Y2Y.plot()

## VIX(변동 지수 %) 시장 불안정성: VIX 지수가 20을 초과하면 일반적으로 시장의 불안정성이 높다고 간주된다. | S&P 500 지수의 연간 변동성을 나타낸다 
VIX = fdr.DataReader('FRED:VIXCLS', start, end)
VIX.index.name = 'Date'
print(VIX)
graph = VIX.plot()

##금융 스트레스 지수
FSI = fdr.DataReader('FRED:STLFSI3', start, end)
FSI.index.name = 'Date'

print(FSI)
graph = FSI.plot()

df.index.name = 'Date'
DGS.index.name = 'Date'
T10Y2Y.index.name = 'Date'
VIX.index.name = 'Date'
FSI.index.name = 'Date'


# 'Date' 인덱스를 datetime 형식으로 변환
adj_close_df.index = pd.to_datetime(adj_close_df.index)
DGS.index = pd.to_datetime(DGS.index)
T10Y2Y.index = pd.to_datetime(T10Y2Y.index)
VIX.index = pd.to_datetime(VIX.index)
FSI.index = pd.to_datetime(FSI.index)
dxy.index = pd.to_datetime(dxy.index)



# 모든 결합된 데이터를 합침
econ_df = adj_close_df.join([DGS, T10Y2Y, VIX], how='left')
print(econ_df)
econ_df.to_csv('econ_data.csv')

