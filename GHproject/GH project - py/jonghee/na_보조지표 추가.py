# pip install ta ㅂ
import pandas as pd
import ta
import plotly.graph_objects as go
import matplotlib.pyplot as plt


df = pd.read_csv('data/new_stock data.csv')
df
# 볼린저 추가
df['BOL_H'] = ta.volatility.bollinger_hband(df['Close'])
df['BOL_AVG'] = ta.volatility.bollinger_mavg(df['Close'])
df['BOL_L'] = ta.volatility.bollinger_lband(df['Close'])

# RSI 추가
df['RSI'] = ta.momentum.rsi(df['Close'])

# MACD 추가
df['MACD'] = ta.trend.macd(df['Close'])
df['MACD_SIGNAL']= ta.trend.macd_signal(df['Close'])

# OBV 추가
df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

df
################################################ Visualization #############################################################
# 볼린저 밴드 시각화
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['BOL_H'], fill=None, mode='lines', line_color='red', name='Bollinger High'))
fig.add_trace(go.Scatter(x=df.index, y=df['BOL_L'], fill='tonexty', mode='lines', fillcolor='#ADD8E6', line_color='blue', name='Bollinger Low'))
fig.add_trace(go.Scatter(x=df.index, y=df['BOL_AVG'], mode='lines', line_color='black', name='Bollinger AVG'))  # 중심선을 BOL_AVG로 수정
fig.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
fig.show()

# RSI 시각화
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
fig_rsi.add_shape(dict(type='line', y0=70, y1=70, x0=df.index[0], x1=df.index[-1], line=dict(color='Red')))
fig_rsi.add_shape(dict(type='line', y0=30, y1=30, x0=df.index[0], x1=df.index[-1], line=dict(color='Green')))
fig_rsi.update_layout(
    title="Relative Strength Index (RSI)", 
    xaxis_title="Date", 
    yaxis_title="RSI Value",
    yaxis=dict(range=[0,100])  # Y축의 범위를 0~100으로 설정
)
fig_rsi.show()

# MACD 시각화
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
fig3.add_trace(go.Scatter(x=df.index, y=df['MACD_SIGNAL'], mode='lines', name='MACD Signal'))
fig3.show()

# 주가와 OBV 시각화
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', yaxis="y2"))
fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], mode='lines', fill='tozeroy', name='OBV'))
fig.update_layout(title="Close Price and OBV", xaxis_title="Date", 
                  yaxis_title="OBV", yaxis2=dict(title="Price", overlaying="y", side="right"))
fig.show()

################################################ Correlation test #############################################################
# 선형관계
correlations = df[['Adj Close', 'BOL_H', 'BOL_AVG', 'BOL_L', 'RSI', 'MACD', 'MACD_SIGNAL', 'OBV']].corr()
print(correlations['Adj Close'])

# 비선형관계 (스피어만 순위 상관계수)
spearman_corr = df[['Adj Close', 'BOL_H', 'BOL_AVG', 'BOL_L', 'RSI', 'MACD', 'MACD_SIGNAL', 'OBV']].corr(method='spearman')
print(spearman_corr['Adj Close'])



# Adj Close와 BOL_H의 관계 시각화
plt.figure(figsize=(12, 6))
plt.plot(df['Adj Close'], label='Adj Close', color='blue')
plt.plot(df['BOL_H'], label='BOL_H', color='red', linestyle='--')
plt.legend()
plt.title('Adj Close vs BOL_H')
plt.show()

# Adj Close와 RSI의 관계 시각화 (산점도)
plt.figure(figsize=(12, 6))
plt.scatter(df['Adj Close'], df['RSI'], alpha=0.5)
plt.title('Scatter plot between Adj Close and RSI')
plt.xlabel('Adj Close')
plt.ylabel('RSI')
plt.show()


from statsmodels.stats.outliers_influence import variance_inflation_factor

# 'Adj Close'를 제외한 변수들만 선택
features = df[['BOL_H', 'BOL_AVG', 'BOL_L', 'RSI', 'MACD', 'MACD_SIGNAL', 'OBV']]
features_with_const = pd.concat([pd.Series(1, index=features.index, name='const'), features], axis=1)

# VIF 계산
vif = pd.Series([variance_inflation_factor(features_with_const.values, i) 
                 for i in range(features_with_const.shape[1])], 
                index=features_with_const.columns)

print(vif)