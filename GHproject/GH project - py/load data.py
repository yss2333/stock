
######################################################## LOAD STOCK DATA ########################################################
import FinanceDataReader as fdr
from datetime import datetime
import pandas as pd

today = datetime.today().strftime('%Y-%m-%d') # 현재 날짜 가져오기

df_krx = fdr.StockListing('KRX')

df = fdr.DataReader('005930','2020-01-02', today) # df = fdr.DataReader('종목코드','시작일자','종료일자')
df

save_path = '/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/data/stock data.csv'  # 파일 저장 경로 설정
df.to_csv(save_path, index=False) # 데이터프레임을 CSV 파일로 저장


######################################################## LOAD Financial Statement Data ########################################################





