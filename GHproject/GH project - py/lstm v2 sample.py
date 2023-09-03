# https://colab.research.google.com/github/teddylee777/machine-learning/blob/master/04-TensorFlow2.0/01-삼성전자-주가예측/02-LSTM-stock-forecasting-with-LSTM-financedatareader.ipynb#scrollTo=fif43Fh-0l4L

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

data = pd.read_csv('data/stock data.csv')
data.head()
data.tail()

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data.index
data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day




plt.figure(figsize=(16, 9))
sns.lineplot(y=data['Close'], x=data.index)
plt.xlabel('time')
plt.ylabel('price')
plt.show()