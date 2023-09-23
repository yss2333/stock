import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Base Model Scaling:
df1 = pd.read_csv('/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/dacon/jonghee_test/Tech_stock_result.csv') # 2023-02-14 ~ 2023-09-08 # 143 prediction
df2 = pd.read_csv('/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/dacon/jonghee_test/industry_result.csv') # 2023-02-14 ~ 2023-09-08 # 143 prediction
df3 = pd.read_csv('/Users/jongheelee/Desktop/JH/personal/GHproject/GH project - py/dacon/jonghee_test/econ_result.csv') # 새로 추가된 데이터

df = pd.merge(df1[['Date', 'Real Price', 'Predicted Price']], 
              df2[['Date', 'Predicted Price']], 
              on='Date', how='inner', suffixes=('_stock', '_fs'))
df = pd.merge(df, df3[['Date', 'Predicted Price']], 
              on='Date', how='inner')
df.columns = ['Date', 'Real Price', 'Tech_Pred', 'Industry_Pred', 'Econ_Pred'] # Rename Column

df1
df2
df3
df

# MinMax
scaler = MinMaxScaler()
scale_cols = ['Real Price', 'Tech_Pred', 'Industry_Pred', 'Econ_Pred']
scaled_df = scaler.fit_transform(df[scale_cols])
scaled_df = pd.DataFrame(scaled_df, columns=scale_cols)

# 2. Create Feature/Label for Stacking model
X_stack = scaled_df[['Tech_Pred', 'Industry_Pred', 'Econ_Pred']].values # Econ_Pred 추가
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
y_val_original = scaler.inverse_transform(np.column_stack([y_val, np.zeros_like(y_val), np.zeros_like(y_val), np.zeros_like(y_val)]))[:, 0]
y_pred_original = scaler.inverse_transform(np.column_stack([y_pred, np.zeros_like(y_pred), np.zeros_like(y_pred), np.zeros_like(y_pred)]))[:, 0]

tech_pred_original = scaler.inverse_transform(np.column_stack([np.zeros_like(y_pred), X_val[:, 0], np.zeros_like(y_pred), np.zeros_like(y_pred)]))[:, 1]
industry_pred_original = scaler.inverse_transform(np.column_stack([np.zeros_like(y_pred), np.zeros_like(y_pred), X_val[:, 1], np.zeros_like(y_pred)]))[:, 2]
econ_pred_original = scaler.inverse_transform(np.column_stack([np.zeros_like(y_pred), np.zeros_like(y_pred), np.zeros_like(y_pred), X_val[:, 2],]))[:, 3]

# 날짜 데이터 추출
date_train, date_val = train_test_split(df['Date'], test_size=0.2, random_state=42)


# 그래프 그리기 준비
plt.figure(figsize=(12, 6))

plot_df = pd.DataFrame({ # 날짜를 정렬하기 위해 DataFrame을 사용
    'Date': date_val,
    'Real Price': y_val_original,
    'Meta Predicted Price': y_pred_original,
    'Technical Predicted Price': tech_pred_original,
    'Industry Predicted Price': industry_pred_original,
    'Econ Predicted Price': econ_pred_original
})

plot_df = plot_df.sort_values(by='Date')  # 날짜로 정렬

plt.plot(plot_df['Date'], plot_df['Real Price'], label='Real Price', linewidth=2)
plt.plot(plot_df['Date'], plot_df['Meta Predicted Price'], label='Meta Predicted Price', linewidth=1.5)
plt.plot(plot_df['Date'], plot_df['Technical Predicted Price'], '--', label='Technical Predicted Price', linewidth=1.5)
plt.plot(plot_df['Date'], plot_df['Industry Predicted Price'], '--', label='Industry Predicted Price', linewidth=1.5)
plt.plot(plot_df['Date'], plot_df['Econ Predicted Price'], '--', label='Econ Predicted Price', linewidth=1.5)

plt.title("Prediction vs Actual Price")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()  # 그래프가 잘 보이도록 레이아웃 조정
plt.show()


##################################################################### 정확도 평가 ##################################################################### 
from sklearn.model_selection import learning_curve

# 학습 곡선 데이터 얻기
train_sizes, train_scores, val_scores = learning_curve(
    meta_model, X_stack, y_stack, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring="neg_mean_squared_error"
)

# 평균 및 표준 편차 계산
train_scores_mean = -train_scores.mean(axis=1)
train_scores_std = train_scores.std(axis=1)
val_scores_mean = -val_scores.mean(axis=1)
val_scores_std = val_scores.std(axis=1)

# 학습 곡선 시각화
plt.figure(figsize=(10, 5))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Mean Squared Error")
plt.legend(loc="best")
plt.title("Learning Curve for Meta Model")
plt.show()


print(scale_cols)
# For tech_pred_original
print(np.column_stack([np.zeros_like(y_pred), X_val[:, 0], np.zeros_like(y_pred), np.zeros_like(y_pred)]).shape)
