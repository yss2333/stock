from functools import reduce
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


tf.keras.backend.clear_session() # 메모리 초기화

# 1. Load Data
Econ = pd.read_csv(f'dacon/심화 loaded data/Econ_data.csv')
Industry = pd.read_csv(f'dacon/심화 loaded data/Industry_data.csv')
Income = pd.read_csv(f'dacon/심화 loaded data/itp_Income.csv')
Cash = pd.read_csv(f'dacon/심화 loaded data/itp_Cash.csv')
Balance = pd.read_csv(f'dacon/심화 loaded data/itp_Balance.csv')
Ratio = pd.read_csv(f'dacon/심화 loaded data/itp_Ratio.csv')

# 1.1. Merge into master table
Econ.set_index('Date', inplace=True) # 'Date' 컬럼을 인덱스로 설정
Industry.set_index('Date', inplace=True)
Income.set_index('Date', inplace=True)
Cash.set_index('Date', inplace=True)
Balance.set_index('Date', inplace=True)
Ratio.set_index('Date', inplace=True)

merged = Econ.drop(columns=['Adj Close'])\
    .merge(Industry.drop(columns=['Adj Close']), left_index=True, right_index=True, how='inner')\
    .merge(Income.drop(columns=['Adj Close']), left_index=True, right_index=True, how='inner')\
    .merge(Cash.drop(columns=['Adj Close']), left_index=True, right_index=True, how='inner')\
    .merge(Balance.drop(columns=['Adj Close']), left_index=True, right_index=True, how='inner')\
    .merge(Ratio.drop(columns=['Adj Close']), left_index=True, right_index=True, how='inner')

merged['Adj Close'] = Econ['Adj Close']  # Adj Close 값 추가 (아무 데이터프레임에서나 가져올 수 있습니다. 여기서는 Econ에서 가져옵니다.)
df = merged

cols = df.columns.tolist() # 컬럼 순서 변경
cols.insert(0, cols.pop(cols.index('Adj Close')))
df = df[cols]

df

## 2.1. Remove Outliers & Missing value
df.isnull().sum() 
df = df.dropna()
df.isnull().sum() 



## 2.2. Normalization - 목적: Gradient Boosting, 시간 단축, 예측력 향상
scaler = MinMaxScaler()
scale_cols = df.columns.tolist()
scaled_df = scaler.fit_transform(df[scale_cols])
scaled_df = pd.DataFrame(scaled_df, columns=scale_cols) 
scaled_df



# Define Input Parameter: feature, label => numpy type
def make_sequene_dataset(feature, label, window_size):
    feature_list = []      # 생성될 feature list
    label_list = []        # 생성될 label list

    for i in range(len(feature)-window_size):
        feature_list.append(feature[i:i+window_size]) # 1-window size까지 feature에 추가 ... 를 반복
        label_list.append(label[i+window_size]) # window size + 1 번째는 label에 추가 ... 를 반복
    return np.array(feature_list), np.array(label_list) # 넘피배열로 변환

# feature_df, label_df 생성
feature_cols = df.columns.drop('Adj Close').tolist()
label_cols = [ 'Adj Close' ]

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

    model.add(LSTM(128, activation='tanh', input_shape=x_train[0].shape, return_sequences=True))  # return_sequences를 True로 설정하여 다음 LSTM 층으로 출력을 전달
    model.add(Dropout(0.2))  

    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.2))  

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
            epochs=100, batch_size=128,        # 100번 학습 - loss가 점점 작아진다, 만약 100번의 학습을 다 하지 않더라도 loss 가 더 줄지 않는다면, 맞춰둔 조건에 따라 조기종료가 이루어진다
            callbacks=[early_stop,  reduceLR]) # save_best_only ,
    
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


# 학습 손실 그래프 그리기
plt.figure(figsize=(15, 10))

for idx, window_size in enumerate(window_sizes):
    plt.subplot(2, 3, idx+1)  # 2x3 grid에서 idx+1 위치에 그래프를 그립니다.
    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.title(f"Training Loss for Window Size {window_size}")
    plt.legend()

plt.tight_layout()  # 각 서브플롯 간격 조절
plt.show()