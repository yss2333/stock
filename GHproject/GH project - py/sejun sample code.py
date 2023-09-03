import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

stocks_tesla = pd.read_csv('data/stock data.csv')
stocks_tesla

plt.figure(figsize=(24, 16))
sns.lineplot(y=stocks_tesla['Volume'], x=stocks_tesla['Date'])
plt.xlabel('time')
plt.ylabel('price')
plt.xticks(rotation=45)

plt.figure(figsize=(16, 9))
sns.lineplot(y=stocks_tesla['Close'], x=stocks_tesla['Date'])
plt.xlabel('time')
plt.ylabel('price')
plt.xticks(rotation=45)

training_set = stocks_tesla.iloc[:800, 1:7].values
test_set = stocks_tesla.iloc[800:, 1:7].values

# 정규화 (normalization)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 60 time-steps and 1 output

X_train = []

y_train = []

for i in range(60, 800):

   X_train.append(training_set_scaled[i-60:i,: ])
   y_train.append(training_set_scaled[i, 4])

X_train, y_train = np.array(X_train), np.array(y_train)

# LSTM 돌릴때 3차원데이터 요구
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 6))

# 50개의 뉴런과 4개의 숨겨진 층으로 LSTM을 만들 것이다
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()

#Adding the first LSTM layer and some Dropout regularisation

model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 6)))
model.add(Dropout(0.2)) # 과적합 방지하기 위한 방법 (20프로)

# Adding a second LSTM layer and some Dropout regularisation

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set

model.fit(X_train, y_train, epochs = 100, batch_size = 32)

training_set  = stocks_tesla.iloc[:800, 1:7]
test_set  = stocks_tesla.iloc[800:, 1:7]


dataset_total = pd.concat((training_set, test_set), axis = 0)

inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values

inputs = sc.transform(inputs)

X_test = []

max_range = len(inputs) - 60
for i in range(60, max_range):

   X_test.append(inputs[i-60:i, :])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 6))

print(X_test.shape)

# (459, 60, 1)

predicted_stock_price = model.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print(predicted_stock_price)