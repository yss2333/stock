import pandas as pd
import matplotlib.pyplot as plt

ticker = 'aapl'


# 콘텐츠 1. FS 변수 고르고 보간법
# 콘텐츠 2. 경제지표 보간법

########################################################################################################################################################################################
## 1. Load data
Income = pd.read_csv(f'dacon/심화 loaded data/{ticker}_FS_Income.csv')
Cash = pd.read_csv(f'dacon/심화 loaded data/{ticker}_FS_Cash.csv')
Balance = pd.read_csv(f'dacon/심화 loaded data/{ticker}_FS_Balance.csv')
Ratio = pd.read_csv(f'dacon/심화 loaded data/{ticker}_FS_Ratio.csv')
Income
# BPS
Balance['BPS'] = Balance['Shareholders\' Equity'] / Income['Shares Outstanding (Basic)']

# PBR
Ratio['PBR'] = Ratio['Market Capitalization'] / (Balance['BPS'] * Income['Shares Outstanding (Basic)'])

# The rest are already provided in the data:
# PER is Ratio['PE Ratio']
# EPS is Income['EPS (Basic)'] and Income['EPS (Diluted)']
# DIV is Ratio['Dividend Yield']

# DPS (This will be an approximation as we're not given dividends directly)
# Assuming market price = Market Capitalization / Shares Outstanding (Basic)
Balance['Market Price'] = Ratio['Market Capitalization'] / Income['Shares Outstanding (Basic)']
Balance['DPS'] = Ratio['Dividend Yield'] * Balance['Market Price']

# Extract the columns
df = pd.DataFrame({
    'Date': Balance['Date'],
    'BPS': Balance['BPS'],
    'PER': Ratio['PE Ratio'],
    'PBR': Ratio['PBR'],
    'EPS': Income['EPS (Diluted)'],
    'DIV': Ratio['Dividend Yield'],
    'DPS': Balance['DPS']
})

df = df.set_index('Date').sort_index()
df.index = pd.to_datetime(df.index)


itp_df = df.resample('D').asfreq() # 일일 데이터로 리샘플링

# 각 변수에 대해 선형 보간법 적용
for column in itp_df.columns:
    itp_df[column] = itp_df[column].interpolate(method='linear')

############ ES
from statsmodels.tsa.holtwinters import ExponentialSmoothing

end_date = '2023-09-08'
forecast_steps = (pd.to_datetime(end_date) - itp_df.index[-1]).days

# 예측을 저장할 데이터프레임 생성
forecast_df = pd.DataFrame(index=pd.date_range(itp_df.index[-1] + pd.Timedelta(days=1), end_date))

# 각 변수에 대해 모델 적합 및 예측
for column in itp_df.columns:
    model = ExponentialSmoothing(itp_df[column], trend='add', seasonal='add', seasonal_periods=365).fit()
    forecast = model.forecast(steps=forecast_steps)
    forecast_df[column] = forecast

# 원래 데이터와 예측값을 합침
result_df = pd.concat([itp_df, forecast_df])

# 결과 출력 (선택적)
print(result_df)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, column in enumerate(df.columns):
    ax = axes[idx]
    df[column].plot(ax=ax, label='Original', linestyle='-', color='blue')
    result_df[column].plot(ax=ax, label='Forecast', linestyle='--', color='red')
    ax.set_title(column)
    ax.legend(loc='best')
    ax.grid(True)

plt.tight_layout()
plt.show()






############

# Add Adj Close
stock = pd.read_csv(f'dacon/심화 loaded data/{ticker}_stock_Tech_data.csv')
stock['Date'] = pd.to_datetime(stock['Date'])
stock.set_index('Date', inplace=True)
stock_selected = stock[['Close']]

df = result_df.merge(stock_selected, left_index=True, right_index=True, how='left')
df
df = df.dropna()
df.isnull().sum() # Now all missing value is dropped

df
save_path = 'dacon/심화 loaded data/FS_summary.csv'  # 파일 저장 경로 설정
df.to_csv(save_path, index=True) # 데이터프레임을 CSV 파일로 저장

###
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



# df.set_index('Date', inplace=True)

df


######################################################################################################################################################################################

######################################################################################################################################################################################

## 2. Data Preprocessing
##   - 1. Remove Outliers & Missing value 
##   - 2. Normalization & Standardization
##   - 3. Define Feature/Label column


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

model.add(LSTM(64, activation='tanh'))

model.add(Dense(1, activation='linear')) # 출력층
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.summary()



# 모델 학습 과정에서의 손실(loss) 값을 기록하기 위한 리스트
train_loss_history = []
val_loss_history = []

# model 학습 (checkpoint, earlystopping, reduceLR 적용)
save_best_only=tf.keras.callbacks.ModelCheckpoint(filepath="jonghee_test/tech lstm_model.h5", monitor='val_loss', save_best_only=True) #가장 좋은 성능을 낸 val_loss가 적은 model만 남겨 놓았습니다.
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10) #검증 손실이 10epoch 동안 좋아지지 않으면 학습률을 0.1 배로 재구성하는 명령어입니다.

hist = model.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          epochs=100, batch_size=16,        # 100번 학습 - loss가 점점 작아진다, 만약 100번의 학습을 다 하지 않더라도 loss 가 더 줄지 않는다면, 맞춰둔 조건에 따라 조기종료가 이루어진다
          callbacks=[early_stop,  reduceLR]) # save_best_only ,

pred = model.predict(x_test)
######################################################################################################################################################################################
# Prediction with Visualization

plt.figure(figsize=(12, 6))
plt.title('Predicted Price Based on FS ratio, window size = 50')
plt.ylabel('Close')
plt.xlabel('period')
plt.plot(y_test, label='actual')
plt.plot(pred, label='prediction')
plt.grid()
plt.legend(loc='best')

plt.show()


###
# 평가지표 2: 학습곡선
train_loss_history.extend(hist.history['loss']) # 학습 과정에서의 손실값(로스) 기록
val_loss_history.extend(hist.history['val_loss'])

plt.figure(figsize=(12, 6))
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.legend()
plt.title('Loss History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
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
dates = df.index[split+window_size:].values

# 결과를 DataFrame으로 변환
result_df = pd.DataFrame({
    'Date': dates,
    'Real Price': real_y_test,
    'Predicted Price': real_pred
})

print(result_df)

save_path = 'dacon/Full test/FS_result.csv'  # 파일 저장 경로 설정
result_df.to_csv(save_path, index=True) # 데이터프레임을 CSV 파일로 저장