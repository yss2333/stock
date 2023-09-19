import pandas as pd
import yfinance as yf

ticker = yf.Ticker("tsla")
info = ticker.info
sector = info.get('sector', None)
sector

df1 = pd.read_csv('sejun/econ_data.csv')
df2 = pd.read_csv('data/GICS_sector.csv')

df2.head()

sector_columns = [col for col in df2.columns if sector in col] + ["Date"]
sector_df = df2[sector_columns]
print(sector_df.head())

merged_df = df1.merge(sector_df, on="Date", how="inner")
print(merged_df.head())



