
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# 1. Base Model Scaling:
df1 = pd.read_csv('dacon/final/Model result/Tech_result.csv') # 2023-02-14 ~ 2023-09-08 # 143 prediction
df2 = pd.read_csv('dacon/final/Model result/Funda_result.csv') # 2023-02-14 ~ 2023-09-08 # 143 prediction

df = pd.merge(df1[['Date', 'Real Price', 'Predicted Price']], 
                     df2[['Date', 'Predicted Price']],
                     on='Date', 
                     how='inner', 
                     suffixes=('_Tech', '_Fund'))

df.columns = ['Date', 'Real Price', 'Tech_Pred', 'Fund_Pred'] # Rename Column
df = df.set_index('Date').sort_index()

df
# MinMax
scaler = MinMaxScaler()
scale_cols = ['Real Price', 'Tech_Pred', 'Fund_Pred']
scaled_df = scaler.fit_transform(df[scale_cols])
scaled_df = pd.DataFrame(scaled_df, columns=scale_cols) 

print(scaled_df)

# 2. Create Feature/Label for Stacking model
X_stack = scaled_df[['Tech_Pred', 'Fund_Pred']].values
y_stack = scaled_df['Real Price'].values

# Data split (20% test)
X_train, X_val, y_train, y_val = train_test_split(X_stack, y_stack, test_size=0.2, random_state=42)




# 3. Meta model training using XGBoost with GridSearchCV
import xgboost as xgb

# 3. Meta model training using XGBoost with GridSearchCV
model = xgb.XGBRegressor(objective='reg:squarederror')


param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'n_estimators': [50, 100],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 5, 10],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1, 1.5]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# Train the model with best parameters
meta_model_xgb = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
meta_model_xgb.fit(X_train, y_train)

# Validate
y_pred = meta_model_xgb.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
##################################### VISUAL #########################################

# 스케일링된 데이터에서 예측값 추출
y_val_original = scaler.inverse_transform(np.column_stack([y_val, np.zeros_like(y_val), np.zeros_like(y_val)]))[:, 0]
y_pred_original = scaler.inverse_transform(np.column_stack([y_pred, np.zeros_like(y_pred), np.zeros_like(y_pred)]))[:, 0]
tech_pred_original = scaler.inverse_transform(np.column_stack([np.zeros_like(y_pred), X_val[:, 0], np.zeros_like(y_pred)]))[:, 1]
fund_pred_original = scaler.inverse_transform(np.column_stack([np.zeros_like(y_pred), np.zeros_like(y_pred), X_val[:, 1]]))[:, 2]

# 날짜 데이터 추출
date_train, date_val = train_test_split(df.index, test_size=0.2, random_state=42)

# 그래프 그리기 준비
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plot_df = pd.DataFrame({
    'Date': date_val,
    'Real Price': y_val_original,
    'Meta Predicted Price': y_pred_original,
    'Stock Predicted Price': tech_pred_original,
    'FS Predicted Price': fund_pred_original
})

# 날짜를 기준으로 정렬
plot_df = plot_df.sort_values(by='Date')
plot_df['Date'] = pd.to_datetime(plot_df['Date'])

plt.figure(figsize=(12, 6))
plt.plot(plot_df['Date'], plot_df['Real Price'], label='Real Price', linewidth=2)
plt.plot(plot_df['Date'], plot_df['Meta Predicted Price'], label='Meta Predicted Price', linewidth=1.5)
plt.plot(plot_df['Date'], plot_df['Stock Predicted Price'], '--', label='Stock Predicted Price', linewidth=1.5)
plt.plot(plot_df['Date'], plot_df['FS Predicted Price'], '--', label='FS Predicted Price', linewidth=1.5)

ax = plt.gca() # X축의 라벨 간격 설정
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 1개월 간격으로 설정
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 날짜 포맷 설정

plt.title("Prediction vs Actual Price")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # X축 라벨 45도로 기울이기
plt.tight_layout()  # 그래프가 잘 보이도록 레이아웃 조정
plt.show()

################################################################################ 정확도 평가 ######################################################################################################### 


# 1. MSE 비교 그래프
y_train_pred = meta_model_xgb.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mse  # 이전에 계산했던 검증 데이터의 MSE

plt.bar(['Train MSE', 'Validation MSE'], [train_mse, val_mse], color=['blue', 'red'])
plt.ylabel('Mean Squared Error')
plt.title('Train vs Validation MSE')
plt.show()


# 2. 교차 검증
cross_val_mse = -cross_val_score(meta_model_xgb, X_stack, y_stack, cv=10, scoring='neg_mean_squared_error').mean()
errors = [mse, cross_val_mse]
labels = ['Test MSE', 'Cross Validation MSE']

plt.bar(labels, errors, color=['blue', 'red'])
plt.ylabel('MSE Value')
plt.title('Comparison of MSE values, fold = 10')
plt.show()



## 3. 학습 곡선 (임의)
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve( # 학습 곡선 데이터 얻기
    meta_model_xgb, X_stack, y_stack, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring="neg_mean_squared_error"
)

train_scores_mean = -train_scores.mean(axis=1) # 평균 및 표준 편차 계산
train_scores_std = train_scores.std(axis=1)
val_scores_mean = -val_scores.mean(axis=1)
val_scores_std = val_scores.std(axis=1)

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


################################################################### 실제 다음날 예측 진행 ###################################################################################################
'''
# Tech와 Fund 모델로부터 얻은 다음 날의 예측 값을 표현하자면 다음과 같습니다:
next_day_tech_pred = tech_predicted_new_original  # 여기에 Tech 모델로부터 얻은 다음 날 예측 값을 넣어주세요.
next_day_fund_pred = fund_predicted_new_original  # 여기에 Fund 모델로부터 얻은 다음 날 예측 값을 넣어주세요.

# 이 값을 스케일링 합니다:
next_day_tech_pred_scaled = scaler.transform([[0, next_day_tech_pred, 0]])[0][1]  # 가격 부분만 스케일링
next_day_fund_pred_scaled = scaler.transform([[0, 0, next_day_fund_pred]])[0][2]  # 가격 부분만 스케일링

# 스택킹 모델을 사용하여 예측합니다:
next_day_meta_pred_scaled = meta_model.predict(np.array([[next_day_tech_pred_scaled, next_day_fund_pred_scaled]]))[0]

# 예측 값을 원래의 스케일로 변환합니다:
next_day_meta_pred = scaler.inverse_transform([[next_day_meta_pred_scaled, 0, 0]])[0][0]

last_date = pd.to_datetime(df.index[-1]) # df의 마지막 행의 날짜를 가져옴

if last_date.weekday() == 4:  # 0: 월요일, 1: 화요일, ..., 4: 금요일
    next_day = last_date + pd.Timedelta(days=3)  # 금요일에서 3일 후는 월요일
else:
    next_day = last_date + pd.Timedelta(days=1)  # 그 외의 경우에는 하루를 더함

print(f"다음 날({next_day.strftime('%Y-%m-%d')})의 Tech 예측 가격은: {tech_predicted_new_original}") 
print(f"다음 날({next_day.strftime('%Y-%m-%d')})의 Funda 예측 가격은: {fund_predicted_new_original}") 
print(f"다음 날({next_day.strftime('%Y-%m-%d')})의 최종 예측 가격은: {next_day_meta_pred}") 
'''

