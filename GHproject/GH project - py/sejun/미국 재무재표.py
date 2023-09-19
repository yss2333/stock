
import yfinance as yf
from pandas_datareader import data as pdr
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

from yahooquery import Ticker

tickers = ['aapl']
for ticker in tickers:
    
    # 분기: 'https://stockanalysis.com/stocks/aapl/financials/?p=quarterly'
    url = "https://stockanalysis.com/stocks/aapl/financials/?p=quarterly".format(ticker)
    
##    # 연간: https://stockanalysis.com/stocks/aapl/financials/quarterly/
##    # url = "https://stockanalysis.com/stocks/{0}/financials/".format(ticker)
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    element_tables = soup.select("table[data-test='financials']")
    # element_tables = soup.select("div[class='overflow-x-auto']")
    # print(element_tables)
    
df = pd.read_html(str(element_tables))[0] #'0번 테이블 뽑기
print(df)

df.to_csv(ticker+'.csv', index=False, encoding='euc-kr')
# 엑셀 파일로 저장하기용
# df.to_excel(ticker+'.xlsx', index=False, encoding='euc-kr')

fs = pd.read_csv(r'C:\Users\yss06\Desktop\python\stock\GHproject\GH project - py\sejun\aapl.csv')
print(fs)