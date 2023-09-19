import yfinance as yf
import pandas as pd
###################################################################################################
start_date = '2020-01-02'
end_date = '2023-09-08'
###################################################################################################


sectors = {
    "VDE": "Energy",              # Vanguard Energy ETF
    "MXI": "Materials",          # iShares Global Materials ETF
    "VIS": "Industrials",        # Vanguard Industrials ETF
    "VCR": "Consumer Discretionary",  # Vanguard Consumer Discretionary ETF
    "XLP": "Consumer Staples",   # Consumer Staples Select Sector SPDR Fund
    "VHT": "Health Care",        # Vanguard Healthcare ETF
    "XLF": "Financials",         # Financial Select Sector SPDR Fund
    "VGT": "Information Technology",  # Vanguard Information Technology ETF
    "VOX": "Communication Services",  # Vanguard Communication Services ETF
    "XLU": "Utilities",          # Utilities Select Sector SPDR Fund
    "VNQ": "Real Estate"         # Vanguard Real Estate Index Fund
}

sector_data = {}

for sector, sector_name in sectors.items():
    data = yf.download(sector, start=start_date, end=end_date)
    # 컬럼 이름 변경
    data.rename(columns={
        'Adj Close': f'{sector_name} Adj Close',
        'Volume': f'{sector_name} Volume'
    }, inplace=True)
    
    sector_data[sector] = data[[f'{sector_name} Adj Close', f'{sector_name} Volume']]

merged_df = pd.concat(sector_data.values(), axis=1)

print(merged_df)


save_path = '/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/data/GICS_sector.csv'  
merged_df.to_csv(save_path, index=True) 

########################################################################################## 비주얼 ####################################################
import matplotlib.pyplot as plt

# 에너지 섹터의 일별 거래량 변화율 추출
it_close = merged_df['Information Technology Adj Close']
it_volume = merged_df['Information Technology Volume']

# 그래프 그리기
plt.figure(figsize=(15, 7))
it_close.plot(title='Information Technology Adj Close', color='blue', grid=True)
plt.ylabel('$')
plt.xlabel('Date')
plt.tight_layout()
plt.show()

# 그래프 그리기
plt.figure(figsize=(15, 7))
it_volume.plot(title='Information Technology Volume', color='blue', grid=True)
plt.ylabel('$')
plt.xlabel('Date')
plt.tight_layout()
plt.show()
