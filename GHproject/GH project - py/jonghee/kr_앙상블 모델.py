import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Base Model Scaling:
df1 = pd.read_csv('data/kr_stock_result.csv') # 2023-02-14 ~ 2023-09-08 # 143 prediction
df2 = pd.read_csv('data/kr_fs_result.csv') # 2023-02-14 ~ 2023-09-08 # 143 prediction

df = pd.merge(df1[['Date', 'Real Price', 'Predicted Price']], 
                     df2[['Date', 'Predicted Price']],
                     on='Date', 
                     how='inner', 
                     suffixes=('_stock', '_fs'))

df.columns = ['Date', 'Real Price', 'Stock_Pred', 'FS_Pred'] # Rename Column
len(df)
df


# MinMax
scaler = MinMaxScaler()
scale_cols = ['Real Price', 'Stock_Pred', 'FS_Pred']
scaled_df = scaler.fit_transform(df[scale_cols])
scaled_df = pd.DataFrame(scaled_df, columns=scale_cols) 

print(scaled_df)

# 2. Create Feature/Label for Stacking model
X_stack = scaled_df[['Stock_Pred', 'FS_Pred']].values
y_stack = scaled_df['Real Price'].values

# Data split (20% test)
X_train, X_val, y_train, y_val = train_test_split(X_stack, y_stack, test_size=0.2, random_state=42)

# 3. Meta model training
meta_model = LinearRegression()
meta_model.fit(X_train, y_train)

# 4. Meta model Predicting

y_pred = meta_model.predict(X_val)


# 5. Test MSE
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error: {mse}")



##################################### VISUAL #########################################

# 스케일링된 데이터에서 예측값 추출
y_val_original = scaler.inverse_transform(np.column_stack([y_val, np.zeros_like(y_val), np.zeros_like(y_val)]))[:, 0]
y_pred_original = scaler.inverse_transform(np.column_stack([y_pred, np.zeros_like(y_pred), np.zeros_like(y_pred)]))[:, 0]
stock_pred_original = scaler.inverse_transform(np.column_stack([np.zeros_like(y_pred), X_val[:, 0], np.zeros_like(y_pred)]))[:, 1]
fs_pred_original = scaler.inverse_transform(np.column_stack([np.zeros_like(y_pred), np.zeros_like(y_pred), X_val[:, 1]]))[:, 2]

# 날짜 데이터 추출
date_train, date_val = train_test_split(df['Date'], test_size=0.2, random_state=42)


# 그래프 그리기 준비
plt.figure(figsize=(12, 6))

plot_df = pd.DataFrame({ # 날짜를 정렬하기 위해 DataFrame을 사용
    'Date': date_val,
    'Real Price': y_val_original,
    'Meta Predicted Price': y_pred_original,
    'Stock Predicted Price': stock_pred_original,
    'FS Predicted Price': fs_pred_original
})

plot_df = plot_df.sort_values(by='Date')  # 날짜로 정렬

plt.plot(plot_df['Date'], plot_df['Real Price'], label='Real Price', linewidth=2)
plt.plot(plot_df['Date'], plot_df['Meta Predicted Price'], label='Meta Predicted Price', linewidth=1.5)
plt.plot(plot_df['Date'], plot_df['Stock Predicted Price'], '--', label='Stock Predicted Price', linewidth=1.5)
plt.plot(plot_df['Date'], plot_df['FS Predicted Price'], '--', label='FS Predicted Price', linewidth=1.5)

plt.title("Prediction vs Actual Price")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()  # 그래프가 잘 보이도록 레이아웃 조정
plt.show()


############################################################### Chart to see each model performance with Chart  ####################################################################################

date_val_idx = date_val.index # 검증 데이터셋의 인덱스 구하기
df_sorted = df.loc[date_val_idx].sort_values(by='Date') # 해당 인덱스 위치에 예측값 추가

len(df_sorted)

################################################################################ 정확도 평가 ######################################################################################################### 
def calculate_mse(real, pred):
    return mean_squared_error(real, pred)

mse_stock = calculate_mse(df_sorted['Real Price'], df_sorted['Stock_Pred'])
mse_fs = calculate_mse(df_sorted['Real Price'], df_sorted['FS_Pred'])
mse_final = calculate_mse(df_sorted['Real Price'], df_sorted['Final_Pred'])

result_table = pd.DataFrame({
    'Prediction Type': ['Stock_Pred', 'FS_Pred', 'Final_Pred'],
    'MSE': [mse_stock, mse_fs, mse_final]
})

print(result_table)


y_train_pred = meta_model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
# 데이터 저장
losses = {
    "train_loss": [mse_train],
    "val_loss": [mse]
}

plt.figure(figsize=(10, 6))
plt.scatter(['Train'], [mse_train], label='Train Loss', s=100)
plt.scatter(['Validation'], [mse], label='Validation Loss', color='red', s=100)
plt.title('Train vs. Validation Loss')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()
