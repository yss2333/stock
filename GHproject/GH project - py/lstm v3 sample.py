'''
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
    
# Set window size
window_size = 50
X, Y = make_sequene_dataset(feature_np, label_np, window_size)
print(X.shape, Y.shape) # (817, 50, 5) (817, 1) ---- batch size, time steps, input dimensions (윈도우 사이즈에 따라, batch size = total sample size - window size)

# Split into train, test (split = int(len(X)*0.95))
split = int(len(X)*0.80) # Recent 200 observations
x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split:]
y_test = Y[split:]

print(x_train.shape, y_train.shape) # (645, 50, 6) (617, 1)
print(x_test.shape, y_test.shape) # (200, 40, 5) (200, 1)


## 4. Construct and Compile model

# model 생성
model = Sequential()

model.add(LSTM(128, # LSTM 계층에 tanh를 activation function으로 가지는 node 수 128개, recurrent_activation은 코드에 명시되지 않았기 때문에 기본값인 'sigmoid'가 Forget 게이트와 Input 게이트의 Relevance 게이트에 사용됩니다.
               activation='tanh', input_shape=x_train[0].shape)) # (40,5) 형태로 들어가게 된다
model.add(Dense(1, activation='linear')) # 출력층
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.summary()



# model 학습 (earlystopping 적용)
early_stop = EarlyStopping(monitor='val_loss', patience=5)

model.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          epochs=100, batch_size=16,        # 100번 학습 - loss가 점점 작아진다, 만약 100번의 학습을 다 하지 않더라도 loss 가 더 줄지 않는다면, 맞춰둔 조건에 따라 조기종료가 이루어진다
          callbacks=[early_stop])

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

# 평균절대값백분율오차계산 (MAPE)
mape = np.sum(abs(y_test - pred) / y_test) / len(x_test)
mae = np.mean(np.abs(y_test - pred))
rmse = np.sqrt(np.mean(np.square(y_test - pred)))

# 지표를 DataFrame으로 만들기
metrics_df = pd.DataFrame({
    'Metrics': ['MAPE', 'MAE', 'RMSE'],
    'Values': [mape, mae, rmse]})

print(metrics_df)

