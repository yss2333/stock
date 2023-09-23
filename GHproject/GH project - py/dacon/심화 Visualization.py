import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import datetime as dt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

## 1. Load data
ticker = 'tsla'

df = pd.read_csv(f'dacon/심화 loaded data/{ticker}_stock_Tech_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

len(df) # 2502

######################################################################################################################################################################################
# 원하는 날짜 구간 입력 받기
start_date = pd.to_datetime(input("Enter the start date (YYYY-MM-DD): "))
end_date = pd.to_datetime(input("Enter the end date (YYYY-MM-DD): "))

# 입력 받은 날짜로 데이터 필터링
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
print(filtered_df)
# 양봉과 음봉 색상 설정
colors = ['red' if close > open else 'blue' for close, open in zip(filtered_df['Close'], filtered_df['Open'])]

# 2x1 서브플롯 생성
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=('Stock Price', 'Volume'))

# 캔들스틱 차트 추가
fig.add_trace(go.Candlestick(x=filtered_df['Date'],
                             open=filtered_df['Open'],
                             high=filtered_df['High'],
                             low=filtered_df['Low'],
                             close=filtered_df['Close'],
                             name='Candlesticks'), row=1, col=1)

# 이평선 추가
fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA5'], mode='lines', name='MA5', line=dict(color='red', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA10'], mode='lines', name='MA10', line=dict(color='blue', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA20'], mode='lines', name='MA20', line=dict(color='purple', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA30'], mode='lines', name='MA30', line=dict(color='green', width=1)), row=1, col=1)

# 거래량 바 추가
fig.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Volume'], name='Volume', marker_color=colors), row=2, col=1)

# 차트 레이아웃 설정
fig.update_layout(title='Stock Price Candlestick Chart with Volume for Selected Dates',
                  xaxis_title='Date',
                  yaxis_title='Stock Price',
                  xaxis_rangeslider_visible=False)

# 차트 보이기
fig.show()
