from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')
# 0. 여기만 입력하세요.
ticker = 'aapl' # 소문자로 입력해야 합니다 아니면 FS 뽑을때 오류
start_date = '2013-09-28'
end_date = today

########################################################### Add Technical Indicator to NA STOCK DATA ########################################################### 

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


########################################################### load Economic Indicator DATA ########################################################### 
adj_close_df = stock_df[['Adj Close']]



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
# FS_Balance['BPS'] = FS_Balance['Shareholders\' Equity'] / FS_Income['Shares Outstanding (Basic)']

# PBR
# FS_Ratio['PBR'] = FS_Ratio['Market Capitalization'] / (FS_Balance['BPS'] * FS_Income['Shares Outstanding (Basic)'])

# PER is Ratio['PE Ratio']
# EPS is Income['EPS (Basic)'] and Income['EPS (Diluted)']
# DIV is Ratio['Dividend Yield']
# DPS is Income['Dividend Per Share']
# EBITDA is Income['EBITDA'] 

FS_Ratio['ROE'] = FS_Income['Net Income'] / FS_Balance['Shareholders\' Equity'] # ROE is Ratio['Return on Equity (ROE)']


FS_Summary = pd.concat([FS_Income, FS_Balance, FS_Ratio, FS_Cash], axis=1)


# FS_Summary = FS_Summary.set_index('Date').sort_index()
FS_Summary.index = pd.to_datetime(FS_Summary.index)
duplicated_columns = FS_Summary.columns[FS_Summary.columns.duplicated()].unique()
FS_Summary = FS_Summary.drop(columns=duplicated_columns)



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
daily_FS_Summary = daily_FS_Summary.drop(columns=['Debt Growth'])




# Add Adj Close
daily_FS_Summary = daily_FS_Summary.merge(adj_close_df, left_index=True, right_index=True, how='left')


daily_FS_Summary = daily_FS_Summary.dropna()
daily_FS_Summary['Date'] = daily_FS_Summary.index
daily_FS_Summary = daily_FS_Summary.reset_index(drop=True)
daily_FS_Summary = daily_FS_Summary.set_index('Date').sort_index()

daily_FS_Summary
# 1.2. Select feature which correlation > 0.6 (한계: 선형 상관계수만 나타냄)
correlation = daily_FS_Summary.corr()['Adj Close']
selected_features = correlation[correlation.abs() > 0.9].index.tolist() # 0.6 이상의 상관계수를 가진 feature들 필터링
selected_features

daily_FS_Summary= daily_FS_Summary[selected_features]
daily_FS_Summary

## 5. Company FS summary daily
save_path = f'dacon/final/Loaded data/{ticker}_FS_summary.csv'  
daily_FS_Summary.to_csv(save_path, index=True) 




###################################################################### 파일저장 ################################################################################################


################################### 메꾼값들 그래프로 비교하기 ##############################3


# 변수를 페이지로 분할
page_size = 9
total_pages = -(-len(FS_Summary.columns) // page_size)

for page in range(total_pages):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # 각 페이지에 해당하는 변수만 그래프로 그리기
    for idx in range(page_size):
        actual_idx = page * page_size + idx
        if actual_idx >= len(FS_Summary.columns): # 마지막 페이지에서 변수가 9개 미만인 경우 방지
            break
        column = FS_Summary.columns[actual_idx]
        
        FS_Summary[column].plot(ax=axes[idx], label='Original', linestyle='-', color='blue')
        daily_FS_Summary[column].plot(ax=axes[idx], label='Forecast', linestyle='--', color='red')
        axes[idx].set_title(column)
        axes[idx].legend(loc='best')
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()
