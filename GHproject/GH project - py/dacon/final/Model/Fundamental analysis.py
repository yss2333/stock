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
ticker = 'nvda'

tf.keras.backend.clear_session() # 메모리 초기화

## 1.1. Load data
df = pd.read_csv(f'dacon/final/Loaded data/{ticker}_FS_summary.csv')
df2 = pd.read_csv(f'dacon/final/Loaded data/Econ_data.csv')
df3 = pd.read_csv(f'dacon/final/Loaded data/Industry_data.csv')


df.set_index('Date', inplace=True)
df2.set_index('Date', inplace=True)
df3.set_index('Date', inplace=True)


# 1.2. Select the required columns from df2
df2_selected = df2[['GDP', 'CPIAUCSL']].copy()
df3_selected = df3[['NDAQ Adj Close', 'DJI Adj Close', 'SPX Adj Close'] + [df3.columns[-2]]]

# 1.3. Merge the selected data with df based on the Date
df = df.merge(df2_selected, on='Date', how='left')
df = df.merge(df3_selected, on='Date', how='left')

'''
features = [col for col in df.columns if col not in ['Adj Close', 'Date']]

n = len(features) # 총 변수 갯수에 따른 행과 열 계산
ncols = 6  # 한 행에 2개의 그래프
nrows = 6
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 2*nrows))
for ax, feature in zip(axes.ravel(), features):
    ax.scatter(df['Adj Close'], df[feature], s=10) # s=10으로 점의 크기 줄임
    ax.set_title(f'Adj Close vs {feature}')
    ax.set_xlabel('Adj Close')
    ax.set_ylabel(feature)
plt.tight_layout()
plt.show()
'''

## 2.1. Remove Outliers & Missing value
df = df.dropna()
df.isnull().sum() 

## 2.2. Normalization
scaler = MinMaxScaler()
scale_cols = df.columns.tolist()
scaled_df = scaler.fit_transform(df[scale_cols])
scaled_df = pd.DataFrame(scaled_df, columns=scale_cols) 

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

print(feature_np.shape, label_np.shape) # (2353, 16) (2353, 1)


## 3. Create data     
## 3. Create data    
# 3.1. Set window size
window_size = 30
X, Y = make_sequene_dataset(feature_np, label_np, window_size)
print(X.shape, Y.shape) # (2452, 50, 5) (2452, 1)

# 3.2. Split into train, test (split = int(len(X)*0.95))
split = int(len(X)*0.80) 
x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split:]
y_test = Y[split:]

print(x_train.shape, y_train.shape) # (1961, 50, 5) (1961, 1)
print(x_test.shape, y_test.shape) # (491, 50, 5) (491, 1)
######################################################################################################################################################################################
## 4. Construct and Compile model

from keras.regularizers import L1L2

model = Sequential()

model.add(LSTM(128, activation='tanh', input_shape=x_train[0].shape, return_sequences=True,
               kernel_regularizer=L1L2(l1=0.01, l2=0.01), recurrent_regularizer=L1L2(l1=0.01, l2=0.01)))
               


model.add(LSTM(64, activation='tanh'))


model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])


model.summary()



# 모델 학습 과정에서의 손실(loss) 값을 기록하기 위한 리스트
train_loss_history = []
val_loss_history = []

# model 학습 (checkpoint, earlystopping, reduceLR 적용)
save_best_only=tf.keras.callbacks.ModelCheckpoint(filepath="jonghee_test/tech lstm_model.h5", monitor='val_loss', save_best_only=True) #가장 좋은 성능을 낸 val_loss가 적은 model만 남겨 놓았습니다.
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
#reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10) #검증 손실이 10epoch 동안 좋아지지 않으면 학습률을 0.1 배로 재구성하는 명령어입니다.

hist = model.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          epochs=100, batch_size=150,        # 100번 학습 - loss가 점점 작아진다, 만약 100번의 학습을 다 하지 않더라도 loss 가 더 줄지 않는다면, 맞춰둔 조건에 따라 조기종료가 이루어진다
          callbacks=[early_stop]) # save_best_only ,

pred = model.predict(x_test)

############################################################################ 평가지표 ##########################################################################################################
# 평가지표 1: 예측 그래프
plt.figure(figsize=(12, 6))
plt.title('Fundamental analysis Prediction model')
plt.ylabel('Close')
plt.xlabel('period')
plt.plot(y_test, label='actual')
plt.plot(pred, label='prediction')
plt.grid()
plt.legend(loc='best')
plt.show()


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


# 평가지표 3: MAPE, MAE, RMSE
mape = np.sum(abs(y_test - pred) / y_test) / len(x_test)
mae = np.mean(np.abs(y_test - pred))
rmse = np.sqrt(np.mean(np.square(y_test - pred)))

metrics_df = pd.DataFrame({
    'Metrics': ['MAPE', 'MAE', 'RMSE'],
    'Values': [mape, mae, rmse]})

print(metrics_df)

#################################################################################### For stacking ####################################################################################

inverse_df = pd.DataFrame(np.zeros((len(y_test), len(scale_cols))), columns=scale_cols) # y_test 역변환을 위한 임시 DataFrame
inverse_df['Adj Close'] = y_test.flatten()
real_y_test = scaler.inverse_transform(inverse_df)[:, inverse_df.columns.get_loc('Adj Close')]

inverse_df['Adj Close'] = pred.flatten() # pred 역변환을 위한 임시 DataFrame
real_pred = scaler.inverse_transform(inverse_df)[:, inverse_df.columns.get_loc('Adj Close')]

dates = df.index[split+window_size:].values # 해당 날짜 가져오기

result_df = pd.DataFrame({
    'Date': dates,
    'Real Price': real_y_test,
    'Predicted Price': real_pred
})

save_path = 'dacon/final/Model result/Funda_result.csv'  # 파일 저장 경로 설정
result_df.to_csv(save_path, index=True) # 데이터프레임을 CSV 파일로 저장


## 진짜 예측값 추출하기

last_date = pd.to_datetime(df.index[-1]) # df의 마지막 행의 날짜를 가져옴

if last_date.weekday() == 4:  # 0: 월요일, 1: 화요일, ..., 4: 금요일
    next_day = last_date + pd.Timedelta(days=3)  # 금요일에서 3일 후는 월요일
else:
    next_day = last_date + pd.Timedelta(days=1)  # 그 외의 경우에는 하루를 더함


# 1. Extract the last 50 days data
latest_data = feature_np[-window_size:]
latest_data = latest_data.reshape(1, window_size, -1)

# 2. Predict the value for the next day
next_day_pred = model.predict(latest_data)

# 3. Inverse transform the predicted value to its original scale
inverse_df_temp = pd.DataFrame(np.zeros((1, len(scale_cols))), columns=scale_cols)  # 역변환을 위한 임시 DataFrame
inverse_df_temp['Adj Close'] = next_day_pred.flatten()
fund_predicted_new_original = scaler.inverse_transform(inverse_df_temp)[:, inverse_df_temp.columns.get_loc('Adj Close')]

# Convert numpy array value to scalar
fund_predicted_new_original = fund_predicted_new_original.item()







    
