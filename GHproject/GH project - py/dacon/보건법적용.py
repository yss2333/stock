import pandas as pd
import matplotlib.pyplot as plt

ticker = 'aapl'

## 1. Load data
Income = pd.read_csv(f'dacon/심화 loaded data/{ticker}_FS_Income.csv')
Cash = pd.read_csv(f'dacon/심화 loaded data/{ticker}_FS_Cash.csv')
Balance = pd.read_csv(f'dacon/심화 loaded data/{ticker}_FS_Balance.csv')
Ratio = pd.read_csv(f'dacon/심화 loaded data/{ticker}_FS_Ratio.csv')


# 변수가 너무 많은 관계로 출처 사이트에서 볼드체 된 변수들만 셀렉
Income_columns = ['Date','Revenue', 'Gross Profit', 'Operating Income', 'Pretax Income', 'Net Income', 
                'Shares Outstanding (Diluted)', 'EPS (Diluted)', 'Free Cash Flow', 'EBITDA', 'EBIT']

Cash_columns = ['Date','Net Income', 'Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Net Cash Flow']

Balance_columns =['Date','Cash & Cash Equivalents', 'Total Current Assets', 'Total Long-Term Assets', 'Total Assets', 
                  'Total Current Liabilities', 'Total Long-Term Liabilities', 'Total Liabilities', 'Total Debt']

income = Income[Income_columns]
cash = Cash[Cash_columns]
balance = Balance[Balance_columns]

income = Income[Income_columns].set_index('Date').sort_index()
cash = Cash[Cash_columns].set_index('Date').sort_index()
balance = Balance[Balance_columns].set_index('Date').sort_index()

income.index = pd.to_datetime(income.index)
cash.index = pd.to_datetime(cash.index)
balance.index = pd.to_datetime(balance.index)


# 보간법
itp_income = income.resample('D').asfreq() # 일일 데이터로 리샘플링
itp_cash = cash.resample('D').asfreq() # 일일 데이터로 리샘플링
itp_balance = balance.resample('D').asfreq() # 일일 데이터로 리샘플링

# 각 변수에 대해 선형 보간법 적용
for column in itp_income.columns:
    itp_income[column] = itp_income[column].interpolate(method='linear')

for column in itp_cash.columns:
    itp_cash[column] = itp_cash[column].interpolate(method='linear')

for column in itp_balance.columns:
    itp_balance[column] = itp_balance[column].interpolate(method='linear')

# Add Adj Close
stock = pd.read_csv(f'dacon/심화 loaded data/{ticker}_stock_Tech_data.csv')
stock['Date'] = pd.to_datetime(stock['Date'])
stock.set_index('Date', inplace=True)
stock_selected = stock[['Adj Close']]

itp_Income_df = itp_income.merge(stock_selected, left_index=True, right_index=True, how='left')
itp_Cash_df = itp_cash.merge(stock_selected, left_index=True, right_index=True, how='left')
itp_Balance_df = itp_balance.merge(stock_selected, left_index=True, right_index=True, how='left')

itp_Income_df = itp_Income_df.dropna()
itp_Income_df.isnull().sum() # Now all missing value is dropped

itp_Cash_df = itp_Cash_df.dropna()
itp_Cash_df.isnull().sum() # Now all missing value is dropped

itp_Balance_df = itp_Balance_df.dropna()
itp_Balance_df.isnull().sum() # Now all missing value is dropped


len(itp_Income_df)
len(itp_Cash_df)
len(itp_Balance_df)

########################### 진짜 테스트
# itp_Income_df, itp_Cash_df, itp_Balance_df의 날짜를 기준으로 병합
df = itp_Income_df.merge(itp_Cash_df, left_index=True, right_index=True, how='inner')
df = df.merge(itp_Balance_df, left_index=True, right_index=True, how='inner')

# 'Adj Close' 컬럼이 중복되면 중복된 컬럼을 삭제
columns_to_drop = [col for col in df.columns if col != 'Adj Close' and col.endswith('Adj Close')]
df.drop(columns=columns_to_drop, inplace=True)

df


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping





## 2.2. Normalization - 목적: Gradient Boosting, 시간 단축, 예측력 향상
scaler = MinMaxScaler()
scale_cols = df.columns.tolist()
scaled_df = scaler.fit_transform(df[scale_cols])
scaled_df = pd.DataFrame(scaled_df, columns=scale_cols) 
scaled_df

# Define Input Parameter: feature, label => numpy type
def make_sequene_dataset(feature, label, window_size):
    feature_list = []      
    label_list = []        
    for i in range(len(feature)-window_size):
        feature_list.append(feature[i:i+window_size]) # 1-window size까지 feature에 추가 ... 를 반복
        label_list.append(label[i+window_size]) # window size + 1 번째는 label에 추가 ... 를 반복
    return np.array(feature_list), np.array(label_list) 

# feature_df, label_df 생성
feature_cols = df.columns.drop('Adj Close').tolist()
label_cols = [ 'Adj Close' ]

feature_df = pd.DataFrame(scaled_df, columns=feature_cols)
label_df = pd.DataFrame(scaled_df, columns=label_cols)


# DataFrame => Numpy 변환
feature_np = feature_df.to_numpy()
label_np = label_df.to_numpy()

print(feature_np.shape, label_np.shape) # (2455, 25) (2483, 1)


## 3. Create data    
# 3.1. Set window size
window_size = 50
X, Y = make_sequene_dataset(feature_np, label_np, window_size)
print(X.shape, Y.shape) # (2452, 50, 5) (2452, 1)

# 3.2. Split into train, test (split = int(len(X)*0.95))
split = int(len(X)*0.80) 
x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split:]
y_test = Y[split:]

print(x_train.shape, y_train.shape) # (1946, 50, 8) (1946, 1)
print(x_test.shape, y_test.shape) # (487, 50, 8) (487, 1)

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


# 모델 학습 과정에서의 손실(loss) 값을 기록하기 위한 리스트
train_loss_history = []
val_loss_history = []

# model 학습 (checkpoint, earlystopping, reduceLR 적용)
# save_best_only=tf.keras.callbacks.ModelCheckpoint(filepath="jonghee_test/price lstm_model.h5", monitor='val_loss', save_best_only=True) #가장 좋은 성능을 낸 val_loss가 적은 model만 남겨 놓았습니다.
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10) #검증 손실이 10epoch 동안 좋아지지 않으면 학습률을 0.1 배로 재구성하는 명령어입니다.

hist = model.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          epochs=100, batch_size=128,        # 100번 학습 - loss가 점점 작아진다, 만약 100번의 학습을 다 하지 않더라도 loss 가 더 줄지 않는다면, 맞춰둔 조건에 따라 조기종료가 이루어진다
          callbacks=[early_stop,  reduceLR]) #save_best_only ,

pred = model.predict(x_test)
######################################################################################################################################################################################
# 학습 과정에서의 손실값(로스) 기록
train_loss_history.extend(hist.history['loss'])
val_loss_history.extend(hist.history['val_loss'])

# 에포크마다 val_loss와 val_mae 출력
for epoch in range(len(hist.history['loss'])):
    print(f"Epoch {epoch+1}/{len(hist.history['loss'])}, Val Loss: {hist.history['val_loss'][epoch]:.4f}, Val MAE: {hist.history['val_mae'][epoch]:.4f}")

# loss와 val_loss 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.legend()
plt.title('Loss History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# Prediction with Visualization

plt.figure(figsize=(12, 6))
plt.title('Predict Adj Close based on Stock Price Only, window_size=50')
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

##########


# y_test 역변환을 위한 임시 DataFrame
inverse_df = pd.DataFrame(np.zeros((len(y_test), len(scale_cols))), columns=scale_cols)
inverse_df['Adj Close'] = y_test.flatten()
real_y_test = scaler.inverse_transform(inverse_df)[:, inverse_df.columns.get_loc('Adj Close')]

# pred 역변환을 위한 임시 DataFrame
inverse_df['Adj Close'] = pred.flatten()
real_pred = scaler.inverse_transform(inverse_df)[:, inverse_df.columns.get_loc('Adj Close')]
# 해당 날짜 가져오기
dates = df['Date'][split+window_size:].values

# 결과를 DataFrame으로 변환
result_df = pd.DataFrame({
    'Date': dates,
    'Real Price': real_y_test,
    'Predicted Price': real_pred
})

print(result_df)

save_path = '/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/jonghee_test/FS_result.csv'  # 파일 저장 경로 설정
result_df.to_csv(save_path, index=True) # 데이터프레임을 CSV 파일로 저장

######################### 스태킹
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Base Model Scaling:
df1 = pd.read_csv('jonghee_test/Tech_stock_result.csv') # 2023-02-14 ~ 2023-09-08 # 143 prediction
df2 = pd.read_csv('jonghee_test/FS_result.csv') # 2023-02-14 ~ 2023-09-08 # 143 prediction

df1
df2








# 그래프로 보간법과 오리지널 데이터 변수별 비교
for column in itp_income.columns:
    plt.figure(figsize=(10,6))
    
    # 원래 값 그리기
    income[column].plot(label=f'Original {column}', grid=True)
    
    # 보간된 값 그리기
    itp_income[column].plot(label=f'Interpolated {column}', linestyle='--')
    
    plt.title(f'Comparison of Original and Interpolated {column}')
    plt.ylabel(column)
    plt.legend()
    
    plt.show()
