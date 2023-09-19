########################################################### Load NA STOCK DATA ########################################################### 
import yfinance as yf
from datetime import datetime

ticker = 'tsla' # 소문자로 입력해야 합니다 아니면 FS 뽑을때 오류
start_date = '2020-01-02'
end_date = '2023-09-08'
# today = datetime.today().strftime('%Y-%m-%d') 

stock_df = yf.download(ticker, start = start_date, end = end_date)
save_path = f'/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/dacon/심화 loaded data/{ticker}_stock_data.csv'  
stock_df.to_csv(save_path, index=True) 
########################################################### Add Technical Indicator to NA STOCK DATA ########################################################### 
import pandas as pd
import ta

###
# 생성되는 변수: 이평선, 볼린저, RSI, MACD, OBV
###

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

save_path = f'/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/dacon/심화 loaded data/{ticker}_Tech_stock_data.csv'  
tech_df.to_csv(save_path, index=True) 

########################################################### load Economic Indicator DATA ########################################################### 
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import numpy as np
import seaborn as sns
import yfinance as yf
from fredapi import Fred

###
# 생성되는 데이터프레임: DXY, DGS, T10Y2Y, VIX, FSI
###

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

##금융 스트레스 지수
FSI = fdr.DataReader('FRED:STLFSI3', start_date, end_date)
FSI.index.name = 'Date'

# 모든 결합된 데이터를 합침
econ_df = adj_close_df.join([DGS, T10Y2Y, VIX, FSI], how='left')

save_path = '/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/dacon/심화 loaded data/Econ_Ind_data.csv'  
econ_df.to_csv(save_path, index=True) 


########################################################### load Industry Indicator DATA ########################################################### 
import yfinance as yf
import pandas as pd

###
# 생성되는 변수: 다우존스지수, S&P500, 나스닥, 러셀2000, etf 기반 산업별 수정종가 변화, etf 기반 산업별 거래량 변화
###


### 태환부분 코드 없음

## ETF 기반 산업별 수정종가 + 거래량 데이터
sectors = {
    "VDE": "Energy",              # Vanguard Energy ETF
    "MXI": "Materials",          # iShares Global Materials ETF
    "VIS": "Industrials",        # Vanguard Industrials ETF
    "VCR": "Consumer Discretionary",  # Vanguard Consumer Discretionary ETF
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

save_path = '/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/dacon/심화 loaded data/GICS_sector.csv'  
ETF.to_csv(save_path, index=True) 

########################################################### load Company Indicator DATA ########################################################### 
import yfinance as yf
from pandas_datareader import data as pdr
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from yahooquery import Ticker

###
# 생성되는 데이터: 티커에 해당하는 FS (~2013.09.28)
###

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
}

url = f"https://stockanalysis.com/stocks/{ticker}/financials/?p=quarterly" 

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'html.parser')
element_tables = soup.select("table[data-test='financials']")

df = pd.read_html(str(element_tables))[0] #'0번 테이블 뽑기
df.to_csv(ticker+'.csv', index=False)

FS_quater = df.transpose()
FS_quater.columns = FS_quater.iloc[0]
FS_quater = df.set_index("Quarter Ended").transpose()
FS_quater.index.name = "Date"
FS_quater.to_csv(ticker+'.csv', index=True, encoding='euc-kr')


save_path = f'/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/dacon/심화 loaded data/{ticker}_FS_quater.csv'  
FS_quater.to_csv(save_path, index=True) 


###################################### Merge into Fundamental analysis data ###########################################

'''
ticker import pandas as pd
import yfinance as yf

yf.Ticker("AAPL")
info = ticker.info
sector = info.get('sector', None)
sector

df1 = pd.read_csv('sejun/econ_data.csv')
df2 = pd.read_csv('data/GICS_sector.csv')

df2.head()

sector_columns = [col for col in df2.columns if sector in col] + ["Date"]
sector_df = df2[sector_columns]
print(sector_df.head())

merged_df = df1.merge(sector_df, on="Date", how="inner")
print(merged_df.head())
'''