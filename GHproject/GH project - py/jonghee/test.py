import os
import pandas as pd
import datetime as dt
from concurrent import futures
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
first_table = tables[0]
second_table = tables[1]

first_table
second_table 

df = first_table
print(df.shape)
df["Symbol"] = df["Symbol"].map(lambda x: x.replace(".", "-"))  # rename symbol to escape symbol error
sp500_tickers = list(df["Symbol"])
df.head()

sectors = df["GICS Sector"].value_counts()
plt.bar(sectors.index, sectors.values)
plt.xticks(rotation=90)
plt.xlabel("sector")
plt.ylabel("number of stocks")
plt.show()

added_year = second_table["Date"]["Date"].map(lambda x: int(x[-4:]))
added_year = added_year.value_counts().sort_index()
plt.bar(added_year.index, added_year)
plt.xticks(rotation=30)
plt.xlabel("year")
plt.ylabel("number of replacesd tickers")
plt.show()