# 0. 여기만 입력하세요.
ticker = 'aapl' # 소문자로 입력해야 합니다 아니면 FS 뽑을때 오류
start_Date = '2013-09-28'
end_Date = '2023-09-08'

########################################################### Add Technical Indicator to NA STOCK DATA ########################################################### 
from Datetime import Datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
from yahooquery import Ticker
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import ta
import FinanceDataReader as fdr
import pandas_datareader as pdr
from pandas_datareader import data as pdr
from fredapi import Fred
import yfinance as yf


# ticker = 'aapl'
#### TECH ####
## 1.1. Load data
stock = pd.read_csv(f'dacon/final/Loaded data/{ticker}_stock_Tech_data.csv')
######################################################## 그래프
df = stock[['Open', 'High', 'Low', 'Close','Adj Close',
              'EMA5','EMA20','EMA60','EMA120','MA5','MA20','MA60','MA120',
              'BOL_H1', 'BOL_AVG', 'BOL_L1','BOL_H2','BOL_L2']]

# 2.1. Check Scatterplot against Adj Close
features = [col for col in df.columns if col not in ['Adj Close', 'Date']]

n = len(features) # 총 변수 갯수에 따른 행과 열 계산
ncols = 6  # 한 행에 2개의 그래프
nrows = 4
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 2*nrows))
for ax, feature in zip(axes.ravel(), features):
    ax.scatter(df['Adj Close'], df[feature], s=10) # s=10으로 점의 크기 줄임
    ax.set_title(f'Adj Close vs {feature}')
    ax.set_xlabel('Adj Close')
    ax.set_ylabel(feature)
plt.tight_layout()
plt.show()


#### 
## 1.1. Load data
df = pd.read_csv(f'dacon/final/Loaded data/{ticker}_FS_summary.csv')
df2 = pd.read_csv(f'dacon/final/Loaded data/Econ_data.csv')
df3 = pd.read_csv(f'dacon/final/Loaded data/Industry_data.csv')


df.set_index('Date', inplace=True)
df2.set_index('Date', inplace=True)
df3.set_index('Date', inplace=True)
df3

# 1.2. Select the required columns from df2
df2_selected = df2[['GDP', 'UNRATE', 'CPIAUCSL']].copy()
df3_selected = df3[['NDAQ Adj Close', 'DJI Adj Close', 'SPX Adj Close'] + [df3.columns[-2]]]

# 1.3. Merge the selected data with df based on the Date
df = df.merge(df2_selected, on='Date', how='left')
df = df.merge(df3_selected, on='Date', how='left')
df2

features = [col for col in df.columns if col not in ['Adj Close', 'Date']]

n = len(features) # 총 변수 갯수에 따른 행과 열 계산
ncols = 5  # 한 행에 2개의 그래프
nrows = 6
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 2*nrows))
for ax, feature in zip(axes.ravel(), features):
    ax.scatter(df['Adj Close'], df[feature], s=10) # s=10으로 점의 크기 줄임
    ax.set_title(f'Adj Close vs {feature}')
    ax.set_xlabel('Adj Close')
    ax.set_ylabel(feature)
plt.tight_layout()
plt.show()

########
df = pd.read_csv(f'dacon/final/Loaded data/{ticker}_FS_Income.csv')
df2 = pd.read_csv(f'dacon/final/Loaded data/{ticker}_FS_Cash.csv')
df3 = pd.read_csv(f'dacon/final/Loaded data/{ticker}_FS_Balance.csv')
df4 = pd.read_csv(f'dacon/final/Loaded data/{ticker}_FS_Ratio.csv')

stock = pd.read_csv(f'dacon/final/Loaded data/{ticker}_stock_Tech_data.csv')

# df, df2, df3, df4의 'Date'를 인덱스로 설정
df.set_index('Date', inplace=True)
df2.set_index('Date', inplace=True)
df3.set_index('Date', inplace=True)
df4.set_index('Date', inplace=True)
stock.set_index('Date', inplace=True)


stock.index = pd.to_datetime(stock.index)
df.index = pd.to_datetime(df.index)
df2.index = pd.to_datetime(df.index)
df3.index = pd.to_datetime(df.index)
df4.index = pd.to_datetime(df.index)

adj_close_df = stock[['Adj Close']]
adj_close_df.index = pd.to_datetime(adj_close_df.index)


itp_df = df.resample('D').asfreq() # 일일 데이터로 리샘플링
for column in itp_df.columns:
    itp_df[column] = itp_df[column].interpolate(method='linear') # 각 변수에 대해 선형 보간법 적용

itp_df2 = df2.resample('D').asfreq() # 일일 데이터로 리샘플링
for column in itp_df2.columns:
    itp_df2[column] = itp_df2[column].interpolate(method='linear') # 각 변수에 대해 선형 보간법 적용
    
itp_df3 = df3.resample('D').asfreq() # 일일 데이터로 리샘플링
for column in itp_df3.columns:
    itp_df3[column] = itp_df3[column].interpolate(method='linear') # 각 변수에 대해 선형 보간법 적용
    
itp_df4 = df4.resample('D').asfreq() # 일일 데이터로 리샘플링
for column in itp_df4.columns:
    itp_df4[column] = itp_df4[column].interpolate(method='linear') # 각 변수에 대해 선형 보간법 적용

# itp_df에 'Adj Close'만 합침
merged_df = itp_df.join(adj_close_df)
merged_df = merged_df.dropna()

merged_df2 = itp_df2.join(adj_close_df)
merged_df2 = merged_df2.dropna()

merged_df3 = itp_df3.join(adj_close_df)
merged_df3 = merged_df3.dropna()

merged_df4 = itp_df4.join(adj_close_df)
merged_df4 = merged_df4.dropna()


# 2.1. Check Scatterplot against Adj Close
features = [col for col in merged_df3.columns if col not in ['Adj Close', 'Date']]

half_len = len(features) // 2
features_first_half = features[:half_len]
features_second_half = features[half_len:]

# 첫 번째 절반의 features 그리기
ncols = 5  # 한 행에 2개의 그래프
nrows = 4

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 2*nrows))
for ax, feature in zip(axes.ravel(), features_first_half):
    ax.scatter(merged_df3['Adj Close'], merged_df3[feature], s=10)
    ax.set_title(f'Adj Close vs {feature}', fontsize=8)
    ax.set_xlabel('Adj Close', fontsize=8)
    ax.set_ylabel(feature, fontsize=8)
plt.tight_layout()
plt.show()

# 두 번째 절반의 features 그리기
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 2*nrows))
for ax, feature in zip(axes.ravel(), features_second_half):
    ax.scatter(merged_df3['Adj Close'], merged_df3[feature], s=10)
    ax.set_title(f'Adj Close vs {feature}', fontsize=8)
    ax.set_xlabel('Adj Close', fontsize=8)
    ax.set_ylabel(feature, fontsize=8)
plt.tight_layout()
plt.show()


correlation = merged_df.corr()['Adj Close']
selected_features = correlation[correlation.abs() > 0.7].index.tolist() # 0.6 이상의 상관계수를 가진 feature들 필터링
selected_features.remove('Adj Close') # 'Adj Close' 제거
print(selected_features)