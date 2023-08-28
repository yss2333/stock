
######################################################## LOAD STOCK DATA ########################################################
import FinanceDataReader as fdr
from datetime import datetime
import pandas as pd

today = datetime.today().strftime('%Y-%m-%d') # 현재 날짜 가져오기

df_krx = fdr.StockListing('KRX')

df = fdr.DataReader('005930','2020-01-02', today) # df = fdr.DataReader('종목코드','시작일자','종료일자')
print(df)

save_path = '/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/data/stock data.csv'  # 파일 저장 경로 설정
df.to_csv(save_path, index=True) # 데이터프레임을 CSV 파일로 저장


######################################################## LOAD Financial Statement Data ########################################################

from pykrx import stock

df2 = stock.get_market_fundamental_by_date("20200102", today, "005930")
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