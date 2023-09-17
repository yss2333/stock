
######################################################## LOAD STOCK DATA ########################################################
import FinanceDataReader as fdr
from datetime import datetime
import pandas as pd

today = datetime.today().strftime('%Y-%m-%d') # 현재 날짜 가져오기

df_krx = fdr.StockListing('KRX')

df = fdr.DataReader('005930','2020-01-02', '2023-09-08') # df = fdr.DataReader('종목코드','시작일자','종료일자')
print(df)

save_path = 'C:\\Users\\yss06\\Desktop\\retire\\dd'  # 파일 저장 경로 설정
df.to_csv(save_path, index=True) # 데이터프레임을 CSV 파일로 저장


######################################################## LOAD Financial Statement Data ########################################################

from pykrx import stock

df2 = stock.get_market_fundamental_by_date("20200102", '20230908', "005930")
print(df2)

save_path = '/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/data/fs data.csv'  # 파일 저장 경로 설정
df2.to_csv(save_path, index=True) # 데이터프레임을 CSV 파일로 저장


######################################################## Merge Data #############################################################################

fs_data = pd.read_csv('/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/data/fs data.csv')
stock_data = pd.read_csv('/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/data/stock data.csv')

# '날짜' 컬럼을 datetime 형식으로 변환
fs_data['날짜'] = pd.to_datetime(fs_data['날짜'])
stock_data['Date'] = pd.to_datetime(stock_data['Date'])


merged_data = pd.merge(stock_data, fs_data, left_on='Date', right_on='날짜', how='inner') # 두 데이터프레임을 '날짜' 컬럼을 기준으로 합치기
merged_data.drop(columns=['날짜'], inplace=True) # 불필요한 컬럼 제거

print(merged_data)

save_path = '/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/data/full data.csv'  # 파일 저장 경로 설정
merged_data.to_csv(save_path, index=True) # 데이터프레임을 CSV 파일로 저장


######################################################## New Stock Data #############################################################################
import yfinance as yf
from datetime import datetime

today = datetime.today().strftime('%Y-%m-%d') # 현재 날짜 가져오기

df = yf.download('AAPL',start = '2020-01-02', end = today)

save_path = '/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/data/new_stock data.csv'  # 파일 저장 경로 설정
df.to_csv(save_path, index=True) # 데이터프레임을 CSV 파일로 저장
##############################################################################################################################
from selenium import webdriver
from selenium.webdriver.common.by import By

# investing.com 의 뉴스기사 URL
news_url = 'https://www.investing.com/news/stock-market-news/arm-shares-open-at-5610-a-share-topping-ipo-price-3174614'

# Chrome driver 설정: 본인의 Chrome버전에 맞는 크롬드라이버가 설치되어있어야함.
driver = webdriver.Chrome()

# URL 요청
driver.get(news_url)

# aritivlePage는 신문기사의 본문
article_page = driver.find_element(By.CLASS_NAME, 'articlePage')

print(article_page.text)