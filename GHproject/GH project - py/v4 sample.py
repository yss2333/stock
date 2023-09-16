'''
# https://github.com/neowizard2018/neowizard/blob/master/TensorFlow2/TF_2_x_LEC_21_LSTM_Example.ipynb
### 시계열 데이터에 생성된 주가 예측하기
### 사용 알고리즘: LSTM
### 수정종가 예측하는것이 목표
'''
'''
## 일반적인 steps
## 1. Load data
## 2. Data Preprocessing
## 3. Create Train data
## 4. Construct LSTM (RNN) structure and train
'''
# Make sure that you have all these libaries available to run the code successfully
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import datetime
from mpl_finance import candlestick2_ohlc
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

## 1. Load data
## 일반적으로 날짜 포함 7개의 칼럼 존재 -> 예측 정확도 상승을 위해 5MA, 10MA 등 이평선 추가

# Load data
df = pd.read_csv('data/stock data.csv')
# Add MA 

df['MA5'] = df['Close'].rolling(window=5).mean()  # 5일 이평선 추가
df['MA10'] = df['Close'].rolling(window=10).mean()  # 10일 이평선 추가
df['MA20'] = df['Close'].rolling(window=20).mean()  # 20일 이평선 추가
df['MA30'] = df['Close'].rolling(window=30).mean()  # 50일 이평선 추가

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df.head()
len(df) # 914

# Data Visualization
# 그래프 구역 나누기
fig = plt.figure(figsize=(20,10))
top_axes = plt.subplot2grid((4,4), (0,0), rowspan=3, colspan=4)
bottom_axes = plt.subplot2grid((4,4), (3,0), rowspan=1, colspan=4, sharex=top_axes)
bottom_axes.get_yaxis().get_major_formatter().set_scientific(False)

# 인덱스 설정
idx = df.index.astype('str')

# 이동평균선 그리기
top_axes.plot(idx, df['MA5'], label='MA5', linewidth=0.7)
top_axes.plot(idx, df['MA10'], label='MA10', linewidth=0.7)
top_axes.plot(idx, df['MA20'], label='MA20', linewidth=0.7)
top_axes.plot(idx, df['MA30'], label='MA30', linewidth=0.7)

# 캔들차트 그리기
candlestick2_ohlc(top_axes, df['Open'], df['High'], 
                  df['Low'], df['Close'],
                  width=0.5, colorup='r', colordown='b')

# 거래량 날짜 지정
color_fuc = lambda x : 'r' if x >= 0 else 'b'
color_list = list(df['Volume'].diff().fillna(0).apply(color_fuc))
bottom_axes.bar(idx, df['Volume'], width=0.5, 
                align='center',
                color=color_list)

# 그래프 title 지정

# X축 티커 숫자 20개로 제한
top_axes.xaxis.set_major_locator(ticker.MaxNLocator(10))
# X축 라벨 지정
bottom_axes.set_xlabel('Date', fontsize=15)

top_axes.legend()
plt.tight_layout()
plt.show()

## 2. Data Preprocessing
##   - 1. Remove Outliers & Missing value 
##   - 2. Normalization & Standardization
##   - 3. Define Feature/Label column

# Basic describe
df.describe()
df.isnull().sum() # Nothing detected, but NaN exists in MA columns

# Remove Missing value 
df = df.dropna()
df.isnull().sum() # Now all missing value is dropped



# Normalization (Date 제외한 모든 수치부분 정규화) - 목적: Gradient Boosting, 시간 단축, 예측력 향상
scaler = MinMaxScaler()
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'MA5', 'MA10', 'MA20', 'MA30']
scaled_df = scaler.fit_transform(df[scale_cols])
scaled_df = pd.DataFrame(scaled_df, columns=scale_cols) 

print(scaled_df)

# Define Input Parameter: feature, label => numpy type
def make_sequene_dataset(feature, label, window_size):
    feature_list = []      # 생성될 feature list
    label_list = []        # 생성될 label list

    for i in range(len(feature)-window_size):
        feature_list.append(feature[i:i+window_size]) # 1-window size까지 feature에 추가 ... 를 반복
        label_list.append(label[i+window_size]) # window size + 1 번째는 label에 추가 ... 를 반복
    return np.array(feature_list), np.array(label_list) # 넘피배열로 변환

# feature_df, label_df 생성
feature_cols = ['MA5', 'MA10', 'MA20', 'MA30', 'Close']
label_cols = [ 'Close' ]

feature_df = pd.DataFrame(scaled_df, columns=feature_cols)
label_df = pd.DataFrame(scaled_df, columns=label_cols)

feature_df
label_df
# DataFrame => Numpy 변환
feature_np = feature_df.to_numpy()
label_np = label_df.to_numpy()

print(feature_np.shape, label_np.shape) # (795, 6) (795, 1)


## 3. Create data
##  - 1. Set window size
##  - 2. Create Train data (As form 3D tensor - include batch size, time steps, input dims)
    
# 결과 저장을 위한 리스트
results = []

# 각 윈도우 사이즈별 예측 결과를 저장하기 위한 리스트
predictions = []
window_sizes = [10, 20, 30, 40, 50, 60]

for window_size in window_sizes:
    # 데이터 생성
    X, Y = make_sequene_dataset(feature_np, label_np, window_size)
    
    # 데이터 분할
    split = int(len(X) * 0.80)
    x_train = X[0:split]
    y_train = Y[0:split]
    x_test = X[split:]
    y_test = Y[split:]
    
    # 모델 생성 및 학습
    model = Sequential()
    model.add(LSTM(128, activation='tanh', input_shape=x_train[0].shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=16, callbacks=[early_stop], verbose=0)
    
    # 예측
    pred = model.predict(x_test)
    predictions.append(pred)
    
    # 평가
    mape = np.mean(abs(y_test - pred) / y_test)
    mae = np.mean(np.abs(y_test - pred))
    rmse = np.sqrt(np.mean(np.square(y_test - pred)))
    
    # 결과 저장
    results.append([window_size, mape, mae, rmse])

# 결과 출력
result_df = pd.DataFrame(results, columns=['Window_Size', 'MAPE', 'MAE', 'RMSE'])
print(result_df)

# 그래프로 결과 비교
plt.figure(figsize=(15, 10))

for idx, window_size in enumerate(window_sizes):
    plt.subplot(2, 3, idx+1)  # 2x3 grid에서 idx+1 위치에 그래프를 그립니다.
    plt.plot(y_test, label='True')
    plt.plot(predictions[idx], label='Predicted')
    plt.title(f"Window Size {window_size}")
    plt.legend()

plt.tight_layout()  # 각 서브플롯 간격 조절
plt.show()