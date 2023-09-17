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

# Load data
df = pd.read_csv('data/full data.csv')
selected_columns = ['Date', 'Close', 'BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS']
df = df[selected_columns]
df

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
df.isnull().sum() # Nothing detected
df.isna().sum()
# Remove Missing value 
df = df.dropna()
df.isnull().sum() # Now all missing value is dropped
len(df)


# Normalization (Date 제외한 모든 수치부분 정규화) - 목적: Gradient Boosting, 시간 단축, 예측력 향상
scaler = MinMaxScaler()
scale_cols = ['Close', 'BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS']
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
feature_cols = ['BPS', 'PER', 'PBR', 'EPS', 'DIV', 'DPS']
label_cols = [ 'Close' ]

feature_df = pd.DataFrame(scaled_df, columns=feature_cols)
label_df = pd.DataFrame(scaled_df, columns=label_cols)

feature_df
label_df
# DataFrame => Numpy 변환
feature_np = feature_df.to_numpy()
label_np = label_df.to_numpy()

print(feature_np.shape, label_np.shape) # (914, 6) (914, 1)

######################################################################################################################################################################################
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

print(x_train.shape, y_train.shape) # (691, 50, 6) (691, 1)
print(x_test.shape, y_test.shape) # (173, 50, 6) (173, 1)

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
plt.title('SAMSUNG FS, window_size=40')
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

#################################################################################### For stacking ####################################################################################
# y_test, pred 값을 역변환하기 위한 임시 DataFrame 생성
inverse_df = pd.DataFrame(np.zeros((len(y_test), len(scale_cols))), columns=scale_cols)
inverse_df['Close'] = y_test.flatten()

# y_test 역변환
real_y_test = scaler.inverse_transform(inverse_df)[:, inverse_df.columns.get_loc('Close')]

# pred 값을 위한 임시 DataFrame 수정
inverse_df['Close'] = pred.flatten()

# pred 역변환
real_pred = scaler.inverse_transform(inverse_df)[:, inverse_df.columns.get_loc('Close')]

# 해당 날짜 가져오기
dates = df['Date'][split+window_size:].values

# 결과를 DataFrame으로 변환
result_df = pd.DataFrame({
    'Date': dates,
    'Real Price': real_y_test,
    'Predicted Price': real_pred
})

print(result_df)

save_path = '/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/data/kr_fs_result.csv'  # 파일 저장 경로 설정
result_df.to_csv(save_path, index=True) # 데이터프레임을 CSV 파일로 저장