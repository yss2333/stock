import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

'''
### 기본적인 방법은 스캐터플랏을 통해 선형적 상관관계 확인 후 상관계수 일정 넘는 변수들로 선택

### 데이터는 올바르게 심화 loaded data에 저장되어있다면 티커만 넣어주면 따로 건드릴것 없다.

1 - 주가데이터 + 기술적 지표
2 - 경제데이터
3 - 산업데이터
4 - 기업데이터 - Income statement
5 - 기업데이터 - Balance Sheet
6 - 기업데이터 - Cash Flow
7 - 기업데이터 - Ratio
'''

ticker = 'aapl'

##### Technical
stock_tech_df = pd.read_csv(f'GHproject\GH project - py\dacon\심화 loaded data\{ticker}_stock_Tech_data.csv')

# 1.1. Check Scatterplot against Adj Close
features = [col for col in stock_tech_df.columns if col not in ['Adj Close', 'Date']]

n = len(features) # Check the number of variables
ncols = 8  # 8 table each row
nrows = int(n / ncols) + (n % ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 2*nrows))
for ax, feature in zip(axes.ravel(), features):
    ax.scatter(stock_tech_df['Adj Close'], stock_tech_df[feature], s=10) # s=10으로 점의 크기 줄임
    ax.set_title(f'Adj Close vs {feature}')
    ax.set_xlabel('Adj Close')
    ax.set_ylabel(feature)
plt.tight_layout()
plt.show()

stock_tech_df['Date'] = pd.to_datetime(stock_tech_df['Date'])  # Convert the 'Date' column to a datetime object
stock_tech_df.set_index('Date', inplace=True)

# 1.2. Select feature which correlation > 0.6 (한계: 선형 상관계수만 나타냄)
correlation = stock_tech_df.corr()['Adj Close']
selected_features = correlation[correlation.abs() > 0.6].index.tolist() # 0.6 이상의 상관계수를 가진 feature들 필터링
selected_features.remove('Adj Close') # 'Adj Close' 제거
print(selected_features) # ['Open', 'High', 'Low', 'Close', 'MA5', 'MA15', 'MA75', 'MA150', 'BOL_H', 'BOL_AVG', 'BOL_L', 'OBV']


###### Fundamental

### Econ
Econ_df = pd.read_csv('dacon/심화 loaded data/Econ_data.csv')

# 2.1. Check Scatterplot against Adj Close
features = [col for col in Econ_df.columns if col not in ['Adj Close', 'Date']]

n = len(features) # 총 변수 갯수에 따른 행과 열 계산
ncols = 3  # 한 행에 2개의 그래프
nrows = int(n / ncols) + (n % ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 2*nrows))
for ax, feature in zip(axes.ravel(), features):
    ax.scatter(Econ_df['Adj Close'], Econ_df[feature], s=10) # s=10으로 점의 크기 줄임
    ax.set_title(f'Adj Close vs {feature}')
    ax.set_xlabel('Adj Close')
    ax.set_ylabel(feature)
plt.tight_layout()
plt.show()

# 2.2. Select feature which correlation > 0.6 (한계: 선형 상관계수만 나타냄)
correlation = Econ_df.corr()['Adj Close']
selected_features = correlation[correlation.abs() > 0.6].index.tolist() # 0.6 이상의 상관계수를 가진 feature들 필터링
selected_features.remove('Adj Close') # 'Adj Close' 제거
print(selected_features) # 0.6 넘는놈 없다....



### Industry
GICS = pd.read_csv('dacon/심화 loaded data/GICS_sector.csv')
Index = pd.read_csv('dacon/심화 loaded data/Index_data.csv')

sector = yf.Ticker(ticker).info.get('sector', None)
sector_columns = [col for col in GICS.columns if sector in col] + ["Date"]
sector_df = GICS[sector_columns]
Industry_df = Index.merge(sector_df, on="Date", how="inner")

# 3.1. Check Scatterplot against Adj Close
features = [col for col in Industry_df.columns if col not in ['Adj Close', 'Date']]

n = len(features) # 총 변수 갯수에 따른 행과 열 계산
ncols = 5  # 한 행에 2개의 그래프
nrows = int(n / ncols) + (n % ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 2*nrows))
for ax, feature in zip(axes.ravel(), features):
    ax.scatter(Industry_df['Adj Close'], Industry_df[feature], s=10) # s=10으로 점의 크기 줄임
    ax.set_title(f'Adj Close vs {feature}')
    ax.set_xlabel('Adj Close')
    ax.set_ylabel(feature)
plt.tight_layout()
plt.show()

# 3.2. Select feature which correlation > 0.6 (한계: 선형 상관계수만 나타냄)
correlation = Industry_df.corr()['Adj Close']
selected_features = correlation[correlation.abs() > 0.6].index.tolist() # 0.6 이상의 상관계수를 가진 feature들 필터링
selected_features.remove('Adj Close') # 'Adj Close' 제거
print(selected_features) # ['DJI Adj Close', 'NDAQ Adj Close', 'SPX Adj Close', 'RUT Adj Close', 'Consumer Cyclical Adj Close']




### Company
FS_Income = pd.read_csv(f'dacon/심화 loaded data/{ticker}_FS_Income.csv')
FS_Balance = pd.read_csv(f'dacon/심화 loaded data/{ticker}_FS_Balance.csv')
FS_Cash = pd.read_csv(f'dacon/심화 loaded data/{ticker}_FS_Cash.csv')
FS_Ratio = pd.read_csv(f'dacon/심화 loaded data/{ticker}_FS_Ratio.csv')

# For graph: 변수가 너무 많은 관계로 출처 사이트에서 볼드체 된 변수들만 셀렉
Income_columns = ['Revenue', 'Gross Profit', 'Operating Income', 'Pretax Income', 'Net Income', 
                'Shares Outstanding (Diluted)', 'EPS (Diluted)', 'Free Cash Flow', 'EBITDA', 'EBIT']

Balance_columns =['Cash & Cash Equivalents', 'Total Current Assets', 'Total Long-Term Assets', 'Total Assets', 
                  'Total Current Liabilities', 'Total Long-Term Liabilities', 'Total Liabilities', 'Total Debt', 'Shareholders Equity']

Cash_columns = ['Net Income', 'Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Net Cash Flow']

# Ratio_columns 는 그냥합시다 

# 4.1. Check Scatterplot against Adj Close
n = len(Income_columns) # 총 변수 갯수에 따른 행과 열 계산
n
ncols = 5  # 한 행에 2개의 그래프
nrows = int(n / ncols) + (n % ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 2*nrows))
for ax, Income_columns in zip(axes.ravel(), Income_columns):
    ax.scatter(FS_Income['Adj Close'], FS_Income[Income_columns], s=10) # s=10으로 점의 크기 줄임
    ax.set_title(f'Adj Close vs {Income_columns}')
    ax.set_xlabel('Adj Close')
    ax.set_ylabel(Income_columns)
plt.tight_layout()
plt.show()

# 4.2. Select all feature which correlation > 0.6 (한계: 선형 상관계수만 나타냄)
correlation = FS_Income.corr()['Adj Close']
selected_features = correlation[correlation.abs() > 0.8].index.tolist() # 0.6 이상의 상관계수를 가진 feature들 필터링
selected_features.remove('Adj Close') # 'Adj Close' 제거
print(selected_features) #['Revenue', 'Cost of Revenue', 'Gross Profit', 'Selling, General & Admin', 'Research & Development', 'Operating Expenses', 'Operating Income', 'Pretax Income', 'Income Tax', 'Net Income', 'Shares Outstanding (Basic)', 'Shares Outstanding (Diluted)', 'EPS (Basic)', 'EPS (Diluted)', 'Operating Cash Flow', 'EBITDA', 'Depreciation & Amortization', 'EBIT']


# 5.1. Check Scatterplot against Adj Close
n = len(Balance_columns) # 총 변수 갯수에 따른 행과 열 계산
n
ncols = 5  # 한 행에 2개의 그래프
nrows = int(n / ncols) + (n % ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 2*nrows))
for ax, Balance_columns in zip(axes.ravel(), Balance_columns):
    ax.scatter(FS_Balance['Adj Close'], FS_Balance[Balance_columns], s=10) # s=10으로 점의 크기 줄임
    ax.set_title(f'Adj Close vs {Balance_columns}')
    ax.set_xlabel('Adj Close')
    ax.set_ylabel(Balance_columns)
plt.tight_layout()
plt.show()

# 5.2. Select all feature which correlation > 0.6 (한계: 선형 상관계수만 나타냄)
correlation = FS_Balance.corr()['Adj Close']
selected_features = correlation[correlation.abs() > 0.8].index.tolist() # 0.6 이상의 상관계수를 가진 feature들 필터링
selected_features.remove('Adj Close') # 'Adj Close' 제거
print(selected_features) ['Cash & Equivalents', 'Cash & Cash Equivalents', 'Receivables', 'Other Current Assets', 'Total Current Assets', 'Property, Plant & Equipment', 'Total Long-Term Assets', 'Total Assets', 'Accounts Payable', 'Deferred Revenue', 'Other Current Liabilities', 'Total Current Liabilities', 'Common Stock', "Shareholders' Equity", 'Net Cash / Debt', 'Net Cash Per Share', 'Working Capital', 'Book Value Per Share']


# 6.1. Check Scatterplot against Adj Close
n = len(Cash_columns) # 총 변수 갯수에 따른 행과 열 계산
n
ncols = 5  # 한 행에 2개의 그래프
nrows = int(n / ncols) + (n % ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 2*nrows))
for ax, Cash_columns in zip(axes.ravel(), Cash_columns):
    ax.scatter(FS_Cash['Adj Close'], FS_Cash[Cash_columns], s=10) # s=10으로 점의 크기 줄임
    ax.set_title(f'Adj Close vs {Cash_columns}')
    ax.set_xlabel('Adj Close')
    ax.set_ylabel(Cash_columns)
plt.tight_layout()
plt.show()

# 6.2. Select all feature which correlation > 0.6 (한계: 선형 상관계수만 나타냄)
correlation = FS_Cash.corr()['Adj Close']
selected_features = correlation[correlation.abs() > 0.6].index.tolist() # 0.6 이상의 상관계수를 가진 feature들 필터링
selected_features.remove('Adj Close') # 'Adj Close' 제거
print(selected_features) # 0.6 넘는놈 없다....


# 7.1. Check Scatterplot against Adj Close
features = [col for col in FS_Ratio.columns if col not in ['Adj Close', 'Date']]

n = len(features) # 총 변수 갯수에 따른 행과 열 계산
n
ncols = 7  # 한 행에 2개의 그래프
nrows = int(n / ncols) + (n % ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 2*nrows))
for ax, feature in zip(axes.ravel(), features):
    ax.scatter(FS_Ratio['Adj Close'], FS_Ratio[feature], s=10) # s=10으로 점의 크기 줄임
    ax.set_title(f'Adj Close vs {feature}')
    ax.set_xlabel('Adj Close')
    ax.set_ylabel(feature)
plt.tight_layout()
plt.show()

# 7.2. Select all feature which correlation > 0.6 (한계: 선형 상관계수만 나타냄)
correlation = FS_Ratio.corr()['Adj Close']
selected_features = correlation[correlation.abs() > 0.6].index.tolist() # 0.6 이상의 상관계수를 가진 feature들 필터링
selected_features.remove('Adj Close') # 'Adj Close' 제거
print(selected_features) # 'Market Capitalization', 'Enterprise Value', 'PS Ratio', 'Debt / Equity Ratio', 'Interest Coverage']