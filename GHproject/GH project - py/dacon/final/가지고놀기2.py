ticker = 'nvda'

'''
시작 금액: 1000불
시작일: 2023-05-10
종료일: 2023-06-30

매커니즘:
1. Technical 예측값과 Fundamental 예측값이 크로스가 나고 Technical이 더 높아진다면 풀매수
2. 반대는 풀매도

반환:
일일 결과에 대한 데이터프레임: 날짜, Predicted Price, Actual, 매수/매도/hold indicator, 거래주식수량(+,-), 거래량(+,-), 보유현금

'''
import pandas as pd

final_df = pd.read_csv(f'/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/dacon/final/Loaded data/{ticker}_가지고놀기.csv')
final_df['Date'] = pd.to_datetime(final_df['Date'])  # 날짜 컬럼을 datetime 타입으로 변환
final_df.head()

def simulate_trading_cross_corrected(df, start_date, end_date):
    # Initial values
    cash = 1000
    shares = 0
    last_action = None  # 'buy', 'sell', or None
    last_stock_pred = None
    last_fs_pred = None

    # Filtering the dataframe based on the start and end date
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    # Creating columns for simulation results
    df['Action'] = 'hold'
    df['Shares'] = 0
    df['Trade Volume'] = 0.0
    df['Cash'] = cash

    for i in range(len(df)):
        today = df.iloc[i]
        stock_pred = today['Meta Predicted Price']
        fs_pred = today['FS Predicted Price']
        today_actual = today['Real Price']

        # Buy condition: If there's a cross and Technical prediction becomes greater than Fundamental prediction
        if last_stock_pred and last_fs_pred and last_stock_pred < last_fs_pred and stock_pred > fs_pred:
            shares_bought = cash // today_actual
            cash -= shares_bought * today_actual

            shares += shares_bought
            last_action = 'buy'

            df.iloc[i, df.columns.get_loc('Action')] = 'buy'
            df.iloc[i, df.columns.get_loc('Shares')] = shares_bought
            df.iloc[i, df.columns.get_loc('Trade Volume')] = shares_bought * today_actual
            df.iloc[i, df.columns.get_loc('Cash')] = cash

        # Sell condition: If there's a cross and Fundamental prediction becomes greater than Technical prediction
        elif last_stock_pred and last_fs_pred and last_stock_pred > last_fs_pred and stock_pred < fs_pred:
            cash += shares * today_actual

            shares_sold = shares  # Selling all shares
            shares = 0
            last_action = 'sell'

            df.iloc[i, df.columns.get_loc('Action')] = 'sell'
            df.iloc[i, df.columns.get_loc('Shares')] = -shares_sold
            df.iloc[i, df.columns.get_loc('Trade Volume')] = -shares_sold * today_actual
            df.iloc[i, df.columns.get_loc('Cash')] = cash

        # Hold condition: No action, just update the cash column
        else:
            df.iloc[i, df.columns.get_loc('Cash')] = cash

        # Update last predictions for the next loop
        last_stock_pred = stock_pred
        last_fs_pred = fs_pred

    # For the last day, if we still have shares, we sell them
    if shares > 0:
        cash += shares * df.iloc[-1]['Real Price']
        df.iloc[-1, df.columns.get_loc('Action')] = 'sell'
        df.iloc[-1, df.columns.get_loc('Shares')] = -shares
        df.iloc[-1, df.columns.get_loc('Trade Volume')] = -shares * df.iloc[-1]['Real Price']
    
    df.iloc[-1, df.columns.get_loc('Cash')] = cash  # Correctly updating cash on the last day

    return df

# Running the simulation with cross logic
result_df_cross_corrected = simulate_trading_cross_corrected(final_df, '2021-10-20', '2023-06-30')
result_df_cross_corrected
