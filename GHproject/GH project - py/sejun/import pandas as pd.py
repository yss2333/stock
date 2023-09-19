import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin
yfin.pdr_override()

# 날짜정하기
sdate = "2020-01-02"
edate = "2023-09-08"
df = pdr.get_data_yahoo("AAPL", start= sdate, end= edate)
df1= pdr.get_data_yahoo("^DJI", start= sdate, end= edate)
df2 = pdr.get_data_yahoo("NDAQ", start= sdate, end= edate)
df3 = pdr.get_data_yahoo("^SPX", start= sdate, end= edate)
df4 = pdr.get_data_yahoo("^RUT", start= sdate, end= edate)


# 데이터 프레임에 이름 다시 정하기
df.rename(columns={'Adj Close': 'Adj Close', 'Volume': 'Volume'}, inplace=True)
df1.rename(columns={'Adj Close': 'DJI Adj Close', 'Volume': 'DJI Volume'}, inplace=True)
df2.rename(columns={'Adj Close': 'NDAQ Adj Close', 'Volume': 'NDAQ Volume'}, inplace=True)
df3.rename(columns={'Adj Close': 'SPX Adj Close', 'Volume': 'SPX Volume'}, inplace=True)
df4.rename(columns={'Adj Close': 'RUT Adj Close', 'Volume': 'RUT Volume'}, inplace=True)

# 칼럼 선택
df = df[['Adj Close', 'Volume']]
df1 = df1[['DJI Adj Close', 'DJI Volume']]
df2 = df2[['NDAQ Adj Close', 'NDAQ Volume']]
df3 = df3[['SPX Adj Close', 'SPX Volume']]
df4 = df4[['RUT Adj Close', 'RUT Volume']]

# Merge
combined_df = pd.concat([df, df1, df2, df3, df4], axis=1, join='outer')
combined_df

# csv
combined_df.to_csv(r'C:\Users\yss06\Desktop\python\stock\GHproject\GH project - py\sejun.csv')
