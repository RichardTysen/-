import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.optimizers import Adam

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
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

# 定义 GRU 模型
def create_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(50, input_shape=input_shape, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

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
    mlp_model.fit(X_train_cv, y_train_cv)
    y_pred_mlp_cv = mlp_model.predict(X_val_cv)
    
    # 重新训练 GRU 模型
    gru_model = create_gru_model((1, X_train_cv.shape[1]))
    gru_model.fit(X_train_cv.reshape((X_train_cv.shape[0], 1, X_train_cv.shape[1])), y_train_cv, epochs=50, batch_size=32, verbose=0)
    y_pred_gru_cv = gru_model.predict(X_val_cv.reshape((X_val_cv.shape[0], 1, X_val_cv.shape[1]))).flatten()
    
    # 优化权重
    for w1 in np.linspace(0, 1, 100):
        w2 = 1 - w1
        y_pred_fused_cv = w1 * y_pred_gru_cv + w2 * y_pred_mlp_cv
        mse_fused_cv = mean_squared_error(y_val_cv, y_pred_fused_cv)
        
        if mse_fused_cv < best_mse:
            best_mse = mse_fused_cv
            best_weights = (w1, w2)

# 使用最优权重进行最终预测
w1, w2 = best_weights

# 重新训练最终的 MLP 和 GRU 模型
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)

gru_model = create_gru_model((1, X_train.shape[1]))
gru_model.fit(X_train.reshape((X_train.shape[0], 1, X_train.shape[1])), y_train, epochs=50, batch_size=32, verbose=0)
y_pred_gru = gru_model.predict(X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))).flatten()

# 融合预测值
y_pred_fused = w1 * y_pred_gru + w2 * y_pred_mlp

# 评估融合模型的性能
mse_fused = mean_squared_error(y_test, y_pred_fused)
mae_fused = mean_absolute_error(y_test, y_pred_fused)
r2_fused = r2_score(y_test, y_pred_fused)

print(f"融合模型的 MSE: {mse_fused}")
print(f"融合模型的 MAE: {mae_fused}")
print(f"融合模型的 R²: {r2_fused}")