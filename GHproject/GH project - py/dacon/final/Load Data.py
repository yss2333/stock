ticker = 'nvda' # 소문자로 입력해야 합니다 아니면 FS 뽑을때 오류
start_date = '2013-09-28'
end_date = '2023-09-09'
########################################################### Add Technical Indicator to NA STOCK DATA ########################################################### 
from datetime import datetime
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

stock_df = yf.download(ticker, start = start_date, end = end_date)

## 이평선 추가
stock_df['MA5'] = stock_df['Close'].rolling(window=5).mean()  # 5일 이평선 추가
stock_df['MA15'] = stock_df['Close'].rolling(window=15).mean()  # 10일 이평선 추가
stock_df['MA75'] = stock_df['Close'].rolling(window=75).mean()  # 20일 이평선 추가
stock_df['MA150'] = stock_df['Close'].rolling(window=150).mean()  # 50일 이평선 추가

## 볼린저 추가
stock_df['BOL_H'] = ta.volatility.bollinger_hband(stock_df['Close'])
stock_df['BOL_AVG'] = ta.volatility.bollinger_mavg(stock_df['Close'])
stock_df['BOL_L'] = ta.volatility.bollinger_lband(stock_df['Close'])

## RSI 추가
stock_df['RSI'] = ta.momentum.rsi(stock_df['Close'])

## MACD 추가
stock_df['MACD'] = ta.trend.macd(stock_df['Close'])
stock_df['MACD_SIGNAL']= ta.trend.macd_signal(stock_df['Close'])

## OBV 추가
stock_df['OBV'] = ta.volume.on_balance_volume(stock_df['Close'], stock_df['Volume'])

tech_df = stock_df

########################################################### load Economic Indicator DATA ########################################################### 
adj_close_df = stock_df[['Adj Close']]

## 미국 달러 환율
dxy = fdr.DataReader('DX-Y.NYB', start_date, end_date)
dxy_series = dxy['Adj Close']
DXY = dxy_series.to_frame(name='DXY.Adj Close') # Convert the Series to a DataFrame

## 미국 국채 금리 (20년, 10년, 5년, 1년)
fred = Fred(api_key = '4c55d0ee6170369793707da4cba1b7be')
dgs2 = fred.get_series('DGS2', observation_start=start_date, observation_end=end_date)
dgs5 = fred.get_series('DGS5', observation_start=start_date, observation_end=end_date)
dgs10 = fred.get_series('DGS10', observation_start=start_date, observation_end=end_date)

DGS = pd.concat([dgs2, dgs5,dgs10], axis=1)
DGS.columns = ['2-year', '5-year', '10-year']
DGS.index.name = 'Date'

## 미국 장단기 금리차 | 금리차가 0에 가까워지거나 음수가 되면 (인버전), 이는 종종 경제의 둔화 또는 경기침체를 앞두고 있다는 시장의 예상을 반영하는 것
T10Y2Y = fdr.DataReader('FRED:T10Y2Y', start_date, end_date)
T10Y2Y.index.name = 'Date'

## VIX(변동 지수 %) 시장 불안정성: VIX 지수가 20을 초과하면 일반적으로 시장의 불안정성이 높다고 간주된다. | S&P 500 지수의 연간 변동성을 나타낸다 
VIX = fdr.DataReader('FRED:VIXCLS', start_date, end_date)
VIX.index.name = 'Date'

'''
##금융 스트레스 지수
FSI = fdr.DataReader('FRED:STLFSI3', start_date, end_date)
FSI.index.name = 'Date'

## GDP - 3달주기
GDP = pd.DataFrame(fred.get_series('GDP',observation_start=start_date, observation_end = end_date),columns=['GDP'])
GDP.index.name = 'Date'
'''

## Unemplotment - 1달주기
Unemployment_Rate = fdr.DataReader('FRED:UNRATE', start_date, end_date)
Unemployment_Rate.index.name = 'Date'

## CPI - 1달주기
CPI = fdr.DataReader('FRED:CPIAUCSL', start_date, end_date)
CPI.index.name = 'Date'

## fedfunds 중앙은행 금리지표 - 1달주기
FEDFUNDS = fdr.DataReader('FRED:FEDFUNDS', start_date, end_date)
FEDFUNDS.index.name = 'Date'


## 월간 데이터들에 대한 보간법 및 Holt-Winters' Exponential Smoothing
econ_month_df = Unemployment_Rate.join([CPI, FEDFUNDS], how='left')
itp_df = econ_month_df.resample('D').asfreq()

last_valid_indices = {column: itp_df[column].last_valid_index() for column in itp_df.columns} # 각 변수의 마지막 유효한 인덱스 찾기
last_valid_indices
for column in itp_df.columns:
    itp_df[column] = itp_df[column].interpolate(method='linear').loc[:last_valid_indices[column]] # 마지막 유효한 인덱스까지만 선형 보간 적용
  
# Holt-Winters' Exponential Smoothing
forecast_steps = (pd.to_datetime(end_date) - itp_df.last_valid_index()).days
forecast_df = pd.DataFrame(index=pd.date_range(itp_df.last_valid_index() + pd.Timedelta(days=1), end_date))

for column in itp_df.columns:   
    data_to_fit = itp_df[column].dropna() # 마지막 유효한 값까지의 데이터만 사용
    model = ExponentialSmoothing(data_to_fit, trend='add', seasonal=None, seasonal_periods=12).fit()
    forecast = model.forecast(steps=forecast_steps)
    forecast_df[column] = forecast

daily_econ = pd.concat([itp_df, forecast_df])

# 모든 결합된 데이터를 합침
econ_df = adj_close_df.join([DGS, T10Y2Y, VIX], how='left')
econ_df.dtypes # Check which Data types are object
econ_df = econ_df.join([daily_econ], how='left')

########################################################### load Industry Indicator DATA ########################################################### 
yf.pdr_override() ## Load 4 index data
df = pdr.get_data_yahoo("^DJI", start=start_date, end=end_date) # 다우지수
df1 = pdr.get_data_yahoo("NDAQ", start=start_date, end=end_date) # 나스닥
df2 = pdr.get_data_yahoo("^SPX", start=start_date, end=end_date) # S&P500
df3 = pdr.get_data_yahoo("^RUT", start=start_date, end=end_date) # 러셀 2000

df.rename(columns={'Adj Close': 'DJI Adj Close', 'Volume': 'DJI Volume'}, inplace=True) # 이름 다시정하기
df1.rename(columns={'Adj Close': 'NDAQ Adj Close', 'Volume': 'NDAQ Volume'}, inplace=True)
df2.rename(columns={'Adj Close': 'SPX Adj Close', 'Volume': 'SPX Volume'}, inplace=True)
df3.rename(columns={'Adj Close': 'RUT Adj Close', 'Volume': 'RUT Volume'}, inplace=True)

df = df[['DJI Adj Close', 'DJI Volume']] # 각 자료 칼럼 선택
df1 = df1[['NDAQ Adj Close', 'NDAQ Volume']]
df2 = df2[['SPX Adj Close', 'SPX Volume']]
df3 = df3[['RUT Adj Close', 'RUT Volume']]

Index_data = pd.concat([df, df1, df2, df3], axis=1, join='outer')
Index_data = Index_data.join(stock_df['Adj Close'], how='left')

## ETF 기반 산업별 수정종가 + 거래량 데이터
sectors = {
    "VDE": "Energy",              # Vanguard Energy ETF
    "MXI": "Materials",          # iShares Global Materials ETF
    "VIS": "Industrials",        # Vanguard Industrials ETF
    "VCR": "Consumer Cyclical",  # Vanguard Consumer Discretionary ETF
    "XLP": "Consumer Staples",   # Consumer Staples Select Sector SPDR Fund
    "VHT": "Health Care",        # Vanguard Healthcare ETF
    "XLF": "Financials",         # Financial Select Sector SPDR Fund
    "VGT": "Information Technology",  # Vanguard Information Technology ETF
    "VOX": "Communication Services",  # Vanguard Communication Services ETF
    "XLU": "Utilities",          # Utilities Select Sector SPDR Fund
    "VNQ": "Real Estate"         # Vanguard Real Estate Index Fund
}
sector_data = {}

for sector, sector_name in sectors.items():
    data = yf.download(sector, start=start_date, end=end_date)
    data.rename(columns={
        'Adj Close': f'{sector_name} Adj Close',
        'Volume': f'{sector_name} Volume'
    }, inplace=True) 
    sector_data[sector] = data[[f'{sector_name} Adj Close', f'{sector_name} Volume']]

ETF = pd.concat(sector_data.values(), axis=1)

## Merge into industry data
sector = yf.Ticker(ticker).info.get('sector', None)
sector_columns = [col for col in ETF.columns if sector in col] 

sector_df = ETF[sector_columns]
Industry_df = Index_data.merge(sector_df, on="Date", how="inner")


########################################################### load Company Indicator DATA ########################################################### 

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

## 1. INCOME STATEMENT
url = f"https://stockanalysis.com/stocks/{ticker}/financials/?p=quarterly" 
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
element_tables = soup.select("table[data-test='financials']")

Income_df = pd.read_html(str(element_tables))[0] #'0번 테이블 뽑기
Income_df.to_csv(ticker+'.csv', index=False)


FS_Income = Income_df.transpose()
FS_Income.columns = FS_Income.iloc[0]
FS_Income = Income_df.set_index("Quarter Ended").transpose()
FS_Income.index.name = "Date"
FS_Income.to_csv(ticker+'.csv', index=True, encoding='euc-kr')
FS_Income = FS_Income.iloc[:-1, :]

for column in FS_Income.columns:
    if FS_Income[column].dtype == 'object':
        FS_Income[column] = FS_Income[column].apply(lambda x: float(x) if '-' in x and x[1:].isdigit() else x) # '-' 뒤에 숫자가 있는 문자열 처리
        if FS_Income[column].dtype == 'object' and FS_Income[column].str.contains('%').any():
            FS_Income[column] = FS_Income[column].apply(lambda x: float(x.replace('%', '')) / 100 if '%' in x else x) # 퍼센트 기호가 있는 문자열 처리
        FS_Income[column] = pd.to_numeric(FS_Income[column], errors='coerce')  # 다른 문자열을 숫자로 변환
        
## 2. RATIO STATEMENT
url = f"https://stockanalysis.com/stocks/{ticker}/financials/ratios/?p=quarterly" 
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
element_tables = soup.select("table[data-test='financials']")

Ratio_df = pd.read_html(str(element_tables))[0] #'0번 테이블 뽑기
Ratio_df.to_csv(ticker+'.csv', index=False)


FS_Ratio = Ratio_df.transpose()
FS_Ratio.columns = FS_Ratio.iloc[0]
FS_Ratio = Ratio_df.set_index("Quarter Ended").transpose()
FS_Ratio.index.name = "Date"
FS_Ratio.to_csv(ticker+'.csv', index=True, encoding='euc-kr')
FS_Ratio = FS_Ratio.iloc[1:-1, :]

for column in FS_Ratio.columns:
    if FS_Ratio[column].dtype == 'object':       
        FS_Ratio[column] = FS_Ratio[column].apply(lambda x: float(x) if '-' in x and x[1:].isdigit() else x)             # '-' 뒤에 숫자가 있는 문자열 처리
        if FS_Ratio[column].dtype == 'object' and FS_Ratio[column].str.contains('%').any():
            FS_Ratio[column] = FS_Ratio[column].apply(lambda x: float(x.replace('%', '')) / 100 if '%' in x else x) # 퍼센트 기호가 있는 문자열 처리       
        FS_Ratio[column] = pd.to_numeric(FS_Ratio[column], errors='coerce') # 다른 문자열을 숫자로 변환

## 3. Balance Sheet
url = f"https://stockanalysis.com/stocks/{ticker}/financials/balance-sheet/?p=quarterly" 
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
element_tables = soup.select("table[data-test='financials']")

Balance_df = pd.read_html(str(element_tables))[0] #'0번 테이블 뽑기
Balance_df.to_csv(ticker+'.csv', index=False)

FS_Balance = Balance_df.transpose()
FS_Balance.columns = FS_Balance.iloc[0]
FS_Balance = Balance_df.set_index("Quarter Ended").transpose()
FS_Balance.index.name = "Date"
FS_Balance.to_csv(ticker+'.csv', index=True, encoding='euc-kr')
FS_Balance = FS_Balance.iloc[:-1, :]

for column in FS_Balance.columns:
    if FS_Balance[column].dtype == 'object':       
        FS_Balance[column] = FS_Balance[column].apply(lambda x: float(x) if '-' in x and x[1:].isdigit() else x)             # '-' 뒤에 숫자가 있는 문자열 처리
        if FS_Balance[column].dtype == 'object' and FS_Balance[column].str.contains('%').any():
            FS_Balance[column] = FS_Balance[column].apply(lambda x: float(x.replace('%', '')) / 100 if '%' in x else x) # 퍼센트 기호가 있는 문자열 처리       
        FS_Balance[column] = pd.to_numeric(FS_Balance[column], errors='coerce') # 다른 문자열을 숫자로 변환

## 4. Cash Flow
url = f"https://stockanalysis.com/stocks/{ticker}/financials/cash-flow-statement/?p=quarterly" 
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
element_tables = soup.select("table[data-test='financials']")

Cash_df = pd.read_html(str(element_tables))[0] #'0번 테이블 뽑기
Cash_df.to_csv(ticker+'.csv', index=False)

FS_Cash = Cash_df.transpose()
FS_Cash.columns = FS_Cash.iloc[0]
FS_Cash = Cash_df.set_index("Quarter Ended").transpose()
FS_Cash.index.name = "Date"
FS_Cash.to_csv(ticker+'.csv', index=True, encoding='euc-kr')
FS_Cash = FS_Cash.iloc[:-1, :]

for column in FS_Cash.columns:
    if FS_Cash[column].dtype == 'object':       
        FS_Cash[column] = FS_Cash[column].apply(lambda x: float(x) if '-' in x and x[1:].isdigit() else x)             # '-' 뒤에 숫자가 있는 문자열 처리
        if FS_Cash[column].dtype == 'object' and FS_Cash[column].str.contains('%').any():
            FS_Cash[column] = FS_Cash[column].apply(lambda x: float(x.replace('%', '')) / 100 if '%' in x else x) # 퍼센트 기호가 있는 문자열 처리       
        FS_Cash[column] = pd.to_numeric(FS_Cash[column], errors='coerce') # 다른 문자열을 숫자로 변환

##############################################################################################################
##############################################################################################################

## Create FS summary table With 보간법 & Holt-Winters' Exponential Smoothing

# BPS
FS_Balance['BPS'] = FS_Balance['Shareholders\' Equity'] / FS_Income['Shares Outstanding (Basic)']

# PBR
FS_Ratio['PBR'] = FS_Ratio['Market Capitalization'] / (FS_Balance['BPS'] * FS_Income['Shares Outstanding (Basic)'])

# PER is Ratio['PE Ratio']
# EPS is Income['EPS (Basic)'] and Income['EPS (Diluted)']
# DIV is Ratio['Dividend Yield']

# DPS (This will be an approximation as we're not given dividends directly)
FS_Balance['Market Price'] = FS_Ratio['Market Capitalization'] / FS_Income['Shares Outstanding (Basic)']  # Assuming market price = Market Capitalization / Shares Outstanding (Basic)
FS_Balance['DPS'] = FS_Ratio['Dividend Yield'] * FS_Balance['Market Price']

# Extract the columns
FS_Summary = pd.DataFrame({
    'Date': FS_Balance.index,
    'BPS': FS_Balance['BPS'],
    'PER': FS_Ratio['PE Ratio'],
    'PBR': FS_Ratio['PBR'],
    'EPS': FS_Income['EPS (Diluted)'],
    'DIV': FS_Ratio['Dividend Yield'],
    'DPS': FS_Balance['DPS']
})

FS_Summary = FS_Summary.set_index('Date').sort_index()
FS_Summary.index = pd.to_datetime(FS_Summary.index)

# 선형보간법
itp_df = FS_Summary.resample('D').asfreq() # 일일 데이터로 리샘플링
for column in itp_df.columns:
    itp_df[column] = itp_df[column].interpolate(method='linear') # 각 변수에 대해 선형 보간법 적용
    
# Holt-Winters' Exponential Smoothing
forecast_steps = (pd.to_datetime(end_date) - itp_df.index[-1]).days
forecast_df = pd.DataFrame(index=pd.date_range(itp_df.index[-1] + pd.Timedelta(days=1), end_date)) # 예측을 저장할 데이터프레임 생성

for column in itp_df.columns:
    model = ExponentialSmoothing(itp_df[column], trend='add', seasonal= None, seasonal_periods=4).fit() # 각 변수에 대해 모델 적합 및 예측
    forecast = model.forecast(steps=forecast_steps)
    forecast_df[column] = forecast

daily_FS_Summary = pd.concat([itp_df, forecast_df])

# Add Adj Close
daily_FS_Summary = daily_FS_Summary.merge(adj_close_df, left_index=True, right_index=True, how='left')
daily_FS_Summary = daily_FS_Summary.dropna()
daily_FS_Summary['Date'] = daily_FS_Summary.index
daily_FS_Summary = daily_FS_Summary.reset_index(drop=True)
daily_FS_Summary = daily_FS_Summary.set_index('Date').sort_index()

###################################################################### 파일저장 ################################################################################################
## 1. Tech indicator
save_path = f'dacon/final/Loaded data/{ticker}_stock_Tech_data.csv'  
tech_df.to_csv(save_path, index=True) 

## 2. Econ indicator
save_path = f'dacon/final/Loaded data/Econ_data.csv'  
econ_df.to_csv(save_path, index=True) 

## 3. Industry indicator
save_path = 'dacon/final/Loaded data/Industry_data.csv'  
Industry_df.to_csv(save_path, index=True) 

## 4. Company indicator
#   save_path = f'dacon/final/Loaded data/{ticker}_FS_Income.csv'  
#   FS_Income.to_csv(save_path, index=True) 

#   save_path = f'dacon/final/Loaded data/{ticker}_FS_Ratio.csv'  
#   FS_Ratio.to_csv(save_path, index=True) 

#    save_path = f'dacon/final/Loaded data/{ticker}_FS_Balance.csv'  
#    FS_Balance.to_csv(save_path, index=True) 

#   save_path = f'dacon/final/Loaded data/{ticker}_FS_Cash.csv'  
#   FS_Cash.to_csv(save_path, index=True) 

## 5. Company FS summary daily
save_path = f'dacon/final/Loaded data/{ticker}_FS_summary.csv'  
daily_FS_Summary.to_csv(save_path, index=True) 

################################### 메꾼값들 그래프로 비교하기 ##############################3
# 1. 경제지표 플랏비교
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 subplot for 4 variables
axes = axes.ravel()

# 'econ_month_df'와 'daily_econ'의 열을 순차적으로 비교
for idx, column in enumerate(econ_month_df.columns):
    ax = axes[idx]
    econ_month_df[column].plot(ax=ax, label='Original', linestyle='-', color='blue')
    daily_econ[column].plot(ax=ax, label='Forecast', linestyle='--', color='red')
    ax.set_title(column)
    ax.legend(loc='best')
    ax.grid(True)
plt.tight_layout()
plt.show()

# 2. 재무제표 Summary 플랏비교
fig, axes = plt.subplots(2, 4, figsize=(15, 10))
axes = axes.ravel()

# 'FS_Summary'와 'daily_FS_Summary'의 열을 순차적으로 비교
for idx, column in enumerate(FS_Summary.columns):
    ax = axes[idx]
    FS_Summary[column].plot(ax=ax, label='Original', linestyle='-', color='blue')
    daily_FS_Summary[column].plot(ax=ax, label='Forecast', linestyle='--', color='red')
    ax.set_title(column)
    ax.legend(loc='best')
    ax.grid(True)
plt.tight_layout()
plt.show()