from IPython.core import history
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.optimizers import Adam
from scipy.stats import randint as sp_randint

# 读取数据
data = pd.read_csv('C:/Users/17869/Desktop/zhengqi_train_processed.txt', delimiter='\t')

# 分离特征和标签
X = data.drop(columns=['target'])
y = data['target']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义 MLP 模型
mlp_model = MLPRegressor(random_state=42)

# 定义 MLP 的超参数网格
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [500, 1000]
}

# 使用 GridSearchCV 进行超参数调优
grid_search_mlp = GridSearchCV(mlp_model, param_grid_mlp, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_mlp.fit(X_train, y_train)
best_mlp_model = grid_search_mlp.best_estimator_
best_mlp_params = grid_search_mlp.best_params_

# 定义 GRU 模型
def create_gru_model(input_shape, units=50, learning_rate=0.001):
    model = Sequential()
    model.add(GRU(units, input_shape=input_shape, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# 定义 GRU 的超参数网格
param_grid_gru = {
    'units': [30, 50, 70],
    'learning_rate': [0.001, 0.0001, 0.01]
}

# 使用 RandomizedSearchCV 进行超参数调优
def tune_gru_model(X_train, y_train, param_grid_gru, n_iter=20):
    best_mse = float('inf')
    best_params = None
    best_model = None
    
    for _ in range(n_iter):
        params = {k: v.rvs() if isinstance(v, sp_randint) else np.random.choice(v) for k, v in param_grid_gru.items()}
        gru_model = create_gru_model((1, X_train.shape[1]), **params)
        gru_model.fit(X_train.reshape((X_train.shape[0], 1, X_train.shape[1])), y_train, epochs=50, batch_size=32, verbose=0)
        y_pred_gru = gru_model.predict(X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))).flatten()
        mse = mean_squared_error(y_train, y_pred_gru)
        
        if mse < best_mse:
            best_mse = mse
            best_params = params
            best_model = gru_model
    
    return best_model, best_params

best_gru_model, best_gru_params = tune_gru_model(X_train, y_train, param_grid_gru)

# 定义 Random Forest 模型
rf_model = RandomForestRegressor(random_state=42)

# 定义 Random Forest 的超参数网格
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 使用 GridSearchCV 进行超参数调优
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
best_rf_params = grid_search_rf.best_params_

# 定义 XGBoost 模型
xgb_model = XGBRegressor(random_state=42)

# 定义 XGBoost 的超参数网格
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 使用 RandomizedSearchCV 进行超参数调优
random_search_xgb = RandomizedSearchCV(xgb_model, param_grid_xgb, n_iter=20, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search_xgb.fit(X_train, y_train)
best_xgb_model = random_search_xgb.best_estimator_
best_xgb_params = random_search_xgb.best_params_

# 定义交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 初始化最优权重和最优 MSE
best_weights = None
best_mse = float('inf')

# 交叉验证优化权重
for train_index, val_index in kf.split(X_train):
    X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
    y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # 重新训练 MLP 模型
    best_mlp_model.fit(X_train_cv, y_train_cv)
    y_pred_mlp_cv = best_mlp_model.predict(X_val_cv)
    
    # 重新训练 GRU 模型
    best_gru_model.fit(X_train_cv.reshape((X_train_cv.shape[0], 1, X_train_cv.shape[1])), y_train_cv, epochs=50, batch_size=32, verbose=0)
    y_pred_gru_cv = best_gru_model.predict(X_val_cv.reshape((X_val_cv.shape[0], 1, X_val_cv.shape[1]))).flatten()
    
    # 重新训练 Random Forest 模型
    best_rf_model.fit(X_train_cv, y_train_cv)
    y_pred_rf_cv = best_rf_model.predict(X_val_cv)
    
    # 重新训练 XGBoost 模型
    best_xgb_model.fit(X_train_cv, y_train_cv)
    y_pred_xgb_cv = best_xgb_model.predict(X_val_cv)
    
    # 优化权重
    for w1 in np.linspace(0, 1, 100):
        for w2 in np.linspace(0, 1, 100):
            for w3 in np.linspace(0, 1, 100):
                w4 = 1 - w1 - w2 - w3
                if w4 >= 0:
                    y_pred_fused_cv = w1 * y_pred_gru_cv + w2 * y_pred_mlp_cv + w3 * y_pred_rf_cv + w4 * y_pred_xgb_cv
                    mse_fused_cv = mean_squared_error(y_val_cv, y_pred_fused_cv)
                    
                    if mse_fused_cv < best_mse:
                        best_mse = mse_fused_cv
                        best_weights = (w1, w2, w3, w4)

# 使用最优权重进行最终预测
w1, w2, w3, w4 = best_weights

# 重新训练最终的 MLP 和 GRU 模型
best_mlp_model.fit(X_train, y_train)
y_pred_mlp = best_mlp_model.predict(X_test)

best_gru_model.fit(X_train.reshape((X_train.shape[0], 1, X_train.shape[1])), y_train, epochs=50, batch_size=32, verbose=0)
y_pred_gru = best_gru_model.predict(X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))).flatten()

# 重新训练最终的 Random Forest 和 XGBoost 模型
best_rf_model.fit(X_train, y_train)
y_pred_rf = best_rf_model.predict(X_test)

best_xgb_model.fit(X_train, y_train)
y_pred_xgb = best_xgb_model.predict(X_test)

# 融合预测值
y_pred_fused = w1 * y_pred_gru + w2 * y_pred_mlp + w3 * y_pred_rf + w4 * y_pred_xgb

# 评估融合模型的性能
mse_fused = mean_squared_error(y_test, y_pred_fused)
mae_fused = mean_absolute_error(y_test, y_pred_fused)
r2_fused = r2_score(y_test, y_pred_fused)

print(f"融合模型的 MSE: {mse_fused}")
print(f"融合模型的 MAE: {mae_fused}")
print(f"融合模型的 R²: {r2_fused}")
print(f"最优权重: {best_weights}")

# 输出最佳超参数
print(f"最佳 MLP 超参数: {best_mlp_params}")
print(f"最佳 GRU 超参数: {best_gru_params}")
print(f"最佳 Random Forest 超参数: {best_rf_params}")
print(f"最佳 XGBoost 超参数: {best_xgb_params}")
