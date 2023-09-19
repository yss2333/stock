'''
# https://github.com/neowizard2018/neowizard/blob/master/TensorFlow2/TF_2_x_LEC_21_LSTM_Example.ipynb
'''
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import datetime
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

## 1. Load data
## 일반적으로 날짜 포함 7개의 칼럼 존재 -> 예측 정확도 상승을 위해 5MA, 10MA 등 이평선 추가

# Load data
df = pd.read_csv('data/new_stock data.csv')
'''
# Add MA 여기선 생략
df['MA5'] = df['Close'].rolling(window=5).mean()  # 5일 이평선 추가
df['MA15'] = df['Close'].rolling(window=15).mean()  # 10일 이평선 추가
df['MA75'] = df['Close'].rolling(window=75).mean()  # 20일 이평선 추가
df['MA150'] = df['Close'].rolling(window=150).mean()  # 50일 이평선 추가
'''
df['Date'] = pd.to_datetime(df['Date'])
#df.set_index('Date', inplace=True)

df.head()
len(df) # 914

######################################################################################################################################################################################

######################################################################################################################################################################################

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
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
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
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
label_cols = [ 'Adj Close' ]

feature_df = pd.DataFrame(scaled_df, columns=feature_cols)
label_df = pd.DataFrame(scaled_df, columns=label_cols)

feature_df
label_df
# DataFrame => Numpy 변환
feature_np = feature_df.to_numpy()
label_np = label_df.to_numpy()

print(feature_np.shape, label_np.shape) # (929, 5) (929, 1)

######################################################################################################################################################################################
## 3. Create data
##  - 1. Set window size
##  - 2. Create Train data (As form 3D tensor - include batch size, time steps, input dims)
    
# Set window size
window_size = 50
X, Y = make_sequene_dataset(feature_np, label_np, window_size)
print(X.shape, Y.shape) # (879, 50, 5) (879, 1) ---- batch size, time steps, input dimensions (윈도우 사이즈에 따라, batch size = total sample size - window size)

# Split into train, test (split = int(len(X)*0.95))
split = int(len(X)*0.80) # Recent 200 observations
x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split:]
y_test = Y[split:]

print(x_train.shape, y_train.shape) # (703, 50, 5) (176, 1)
print(x_test.shape, y_test.shape) # (176, 50, 5) (176, 1)

######################################################################################################################################################################################
## 4. Construct and Compile model

# model 생성
model = Sequential()

model.add(LSTM(128, activation='tanh', input_shape=x_train[0].shape, return_sequences=True))  # return_sequences를 True로 설정하여 다음 LSTM 층으로 출력을 전달
model.add(Dropout(0.2))  

model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.2))  

model.add(Dense(1, activation='linear')) # 출력층
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.summary()



# model 학습 (earlystopping 적용)
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# model checkpoint 추가?

model.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          epochs=100, batch_size=16,        # 100번 학습 - loss가 점점 작아진다, 만약 100번의 학습을 다 하지 않더라도 loss 가 더 줄지 않는다면, 맞춰둔 조건에 따라 조기종료가 이루어진다
          callbacks=[early_stop])
######################################################################################################################################################################################
# Prediction with Visualization
pred = model.predict(x_test)

plt.figure(figsize=(12, 6))
plt.title('3MA + 5MA + Close, window_size=40')
plt.ylabel('Close')
plt.xlabel('period')
plt.plot(y_test, label='actual')
plt.plot(pred, label='prediction')
plt.grid()
plt.legend(loc='best')

plt.show()
######################################################################################################################################################################################
# 평균절대값백분율오차계산 (MAPE)
mape = np.sum(abs(y_test - pred) / y_test) / len(x_test)
mae = np.mean(np.abs(y_test - pred))
rmse = np.sqrt(np.mean(np.square(y_test - pred)))

# 지표를 DataFrame으로 만들기
metrics_df = pd.DataFrame({
    'Metrics': ['MAPE', 'MAE', 'RMSE'],
    'Values': [mape, mae, rmse]})

print(metrics_df)

