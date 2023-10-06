ticker = 'nvda'

'''
시작 금액: 1000불
시작일: 2023-05-10
종료일: 2023-06-30

매커니즘:
1. 다음날 예측가격이 오늘 실제가격보다 1프로 높으면 풀매수
2. 구매당시 가격보다 오늘 실제가격이 1프로 높다면 (예측이잘됐다면) 풀매도
3. 홀드는 이어지다가 2번 조건만족되면 풀매도
4. 1~3번 조건 반복

반환:
일일 결과에 대한 데이터프레임: 날짜, Predicted Price, Actual, 매수/매도/hold indicator, 거래주식수량(+,-), 거래량(+,-), 보유현금

'''
import pandas as pd


final_df = pd.read_csv(f'/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/dacon/final/Loaded data/{ticker}_가지고놀기.csv')
final_df['Date'] = pd.to_datetime(final_df['Date'])  # 날짜 컬럼을 datetime 타입으로 변환
final_df.head()

start_date = '2023-04-20', 
end_date = '2023-09-01'

df = final_df[(final_df['Date'] >= start_date) & (final_df['Date'] <= end_date)].copy()




def simulate_trading_corrected(df, start_date, end_date):
    # Initial values
    cash = 1000
    shares = 0
    purchase_price = None

    # Filtering the dataframe based on the start and end date
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    # Creating columns for simulation results
    df['Action'] = 'hold'
    df['Shares'] = 0
    df['Trade Volume'] = 0.0
    df['Cash'] = cash

    for i in range(len(df) - 1):  # -1 because we're comparing with the next day's prediction
        today_actual = df.iloc[i]['Actual']
        tomorrow_predicted = df.iloc[i+1]['Predicted Price']

        # Buy condition: If tomorrow's predicted price is > 1% of today's actual price, we have cash, and we can afford at least one share
        if tomorrow_predicted > today_actual * 1.01 and cash > 0 and cash >= today_actual:
            shares_bought = cash // today_actual
            cash -= shares_bought * today_actual

            shares += shares_bought
            purchase_price = today_actual

            df.iloc[i, df.columns.get_loc('Action')] = 'buy'
            df.iloc[i, df.columns.get_loc('Shares')] = shares_bought
            df.iloc[i, df.columns.get_loc('Trade Volume')] = shares_bought * today_actual
            df.iloc[i, df.columns.get_loc('Cash')] = cash

        # Sell condition: If today's actual price is > 1% of purchase price and we have shares
        elif purchase_price and today_actual > purchase_price * 1.01 and shares > 0:
            cash += shares * today_actual

            shares_sold = shares  # Selling all shares
            shares = 0
            purchase_price = None

            df.iloc[i, df.columns.get_loc('Action')] = 'sell'
            df.iloc[i, df.columns.get_loc('Shares')] = -shares_sold
            df.iloc[i, df.columns.get_loc('Trade Volume')] = -shares_sold * today_actual
            df.iloc[i, df.columns.get_loc('Cash')] = cash

        # Hold condition: No action, just update the cash column
        else:
            df.iloc[i, df.columns.get_loc('Cash')] = cash

    # For the last day, if we still have shares, we sell them
    if shares > 0:
        cash += shares * df.iloc[-1]['Actual']
        df.iloc[-1, df.columns.get_loc('Action')] = 'sell'
        df.iloc[-1, df.columns.get_loc('Shares')] = -shares
        df.iloc[-1, df.columns.get_loc('Trade Volume')] = -shares * df.iloc[-1]['Actual']
    
    df.iloc[-1, df.columns.get_loc('Cash')] = cash  # Correctly updating cash on the last day

    return df

# Running the corrected simulation function for the provided sample data
result_df_corrected = simulate_trading_corrected(final_df, '2023-04-20', '2023-09-01')
result_df_corrected
