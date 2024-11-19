import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('C:/Users/17869/Desktop/zhengqi_train_processed.txt', delimiter='\t')
X = data.drop(columns=['target']) 
y = data['target']
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 定义基模型
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
gru_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_regressor = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 训练基模型
mlp_regressor.fit(X_train, y_train)
gru_regressor.fit(X_train, y_train)
rf_regressor.fit(X_train, y_train)
xgb_regressor.fit(X_train, y_train)

# 预测
y_pred_mlp = mlp_regressor.predict(X_test)
y_pred_gru = gru_regressor.predict(X_test)
y_pred_rf = rf_regressor.predict(X_test)
y_pred_xgb = xgb_regressor.predict(X_test)

# 加权平均预测结果
weights = [0.2, 0.3, 0.25, 0.25]  # 权重可以根据需要调整
y_pred_weighted_avg = (
    weights[0] * y_pred_mlp +
    weights[1] * y_pred_gru +
    weights[2] * y_pred_rf +
    weights[3] * y_pred_xgb
)

# 投票预测结果（取平均值）
y_pred_voting_avg = (y_pred_mlp + y_pred_gru + y_pred_rf + y_pred_xgb) / 4

# 创建堆叠回归器
stacking_regressor = StackingRegressor(
    estimators=[
        ('mlp', mlp_regressor),
        ('gru', gru_regressor),
        ('rf', rf_regressor),
        ('xgb', xgb_regressor)
    ],
    final_estimator=RandomForestRegressor(n_estimators=100, random_state=42),
    cv=5
)

stacking_regressor.fit(X_train, y_train)
y_pred_stacking = stacking_regressor.predict(X_test)
y_pred_hybrid = (y_pred_weighted_avg + y_pred_voting_avg + y_pred_stacking) / 3

# 评估模型性能
mse = mean_squared_error(y_test, y_pred_hybrid)
mae = mean_absolute_error(y_test, y_pred_hybrid)
r2 = r2_score(y_test, y_pred_hybrid)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')



