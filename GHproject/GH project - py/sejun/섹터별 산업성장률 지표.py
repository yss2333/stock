import yfinance as yf
import pandas as pd

sectors = {
    "XLK": "정보기술",
    "XLY": "필수소비재",
    "XLC": "통신 서비스",
    "XLE": "에너지",
    "XLF": "금융",
    "XLV": "건강관리",
    "XLI": "산업재",
    "XLB": "소재",
    "XLRE": "부동산",
    "XLG": "주식",
    "XLU": "공공재"
}

start_date = '2020-01-02'
end_date = '2023-09-08'

sector_data = {}


for sector, sector_name in sectors.items():
    data = yf.download(sector, start=start_date, end=end_date)
    data['Daily Returns'] = data['Adj Close'].pct_change() * 100
    
    # 컬럼 이름 변경
    data.rename(columns={
        'Adj Close': f'{sector_name} Adj Close',
        'Daily Returns': f'{sector_name} Daily Returns'
    }, inplace=True)
    
    sector_data[sector] = data[[f'{sector_name} Adj Close', f'{sector_name} Daily Returns']]

merged_df = pd.concat(sector_data.values(), axis=1)

print(merged_df)

merged_df.to_csv('GICS.csv')







for sector in sectors:
    data = yf.download(sector, start=start_date, end=end_date)
    data['Daily Returns'] = data['Adj Close'].pct_change() * 100
    sector_data[sector] = data[['Adj Close', 'Daily Returns']]
    
for sector, data in sector_data.items():
    print(f"\n{sector} Data:")
    print(data.head())   


