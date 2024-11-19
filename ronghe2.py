import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 加载数据
data = pd.read_csv('C:/Users/17869/Desktop/zhengqi_train_processed.txt', delimiter='\t')

# 分离特征和目标变量
X = data.drop(columns=['target'])  # 假设目标变量名为'target'
y = data['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义基模型
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
gru_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_regressor = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 创建投票回归器
voting_regressor = VotingRegressor(
    estimators=[
        ('mlp', mlp_regressor),
        ('gru', gru_regressor),
        ('rf', rf_regressor),
        ('xgb', xgb_regressor)
    ]
)

# 训练投票回归器
voting_regressor.fit(X_train, y_train)

# 预测
y_pred = voting_regressor.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')