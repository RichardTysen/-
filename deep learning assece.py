import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('C:/Users/17869/Desktop/zhengqi_train_processed.txt', delimiter='\t')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 标准化数据
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 设置交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def build_mlp_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    return model

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(32, input_shape=input_shape, activation='relu'))
    model.add(Dense(1))
    return model

def build_simple_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(32, input_shape=input_shape, activation='relu'))
    model.add(Dense(1))
    return model

def evaluate_mlp_model(input_shape, X, y, kf):
    mse_scores = []
    mae_scores = []
    r2_scores = []

    for _ in range(3):  # 三次循环
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = build_mlp_model(input_shape)
            model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, callbacks=[early_stopping], verbose=0)

            y_pred = model.predict(X_test)
            y_pred_original = scaler_y.inverse_transform(y_pred).flatten()
            y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

            mse = mean_squared_error(y_test_original, y_pred_original)
            mae = mean_absolute_error(y_test_original, y_pred_original)
            r2 = r2_score(y_test_original, y_pred_original)

            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)

    mse_avg = np.mean(mse_scores)
    mae_avg = np.mean(mae_scores)
    r2_avg = np.mean(r2_scores)

    return mse_avg, mae_avg, r2_avg

def evaluate_single_model(model_builder, input_shape, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(input_shape) == 2:
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    model = model_builder(input_shape)
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, callbacks=[early_stopping], verbose=0)

    y_pred = model.predict(X_test)
    y_pred_original = scaler_y.inverse_transform(y_pred).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)

    return mse, mae, r2

# 评估 MLP 模型
mlp_input_shape = X.shape[1]
mlp_mse, mlp_mae, mlp_r2 = evaluate_mlp_model(mlp_input_shape, X, y, kf)

# 评估 GRU 模型
gru_input_shape = (1, X.shape[1])
gru_mse, gru_mae, gru_r2 = evaluate_single_model(build_gru_model, gru_input_shape, X, y)

# 评估 SimpleRNN 模型
simple_rnn_input_shape = (1, X.shape[1])
simple_rnn_mse, simple_rnn_mae, simple_rnn_r2 = evaluate_single_model(build_simple_rnn_model, simple_rnn_input_shape, X, y)

# 打印结果
print("MLP 均方误差（MSE）平均：", mlp_mse)
print("MLP 平均绝对误差（MAE）平均：", mlp_mae)
print("MLP 决定系数（R²）平均：", mlp_r2)

print("GRU 均方误差（MSE）：", gru_mse)
print("GRU 平均绝对误差（MAE）：", gru_mae)
print("GRU 决定系数（R²）：", gru_r2)

print("SimpleRNN 均方误差（MSE）：", simple_rnn_mse)
print("SimpleRNN 平均绝对误差（MAE）：", simple_rnn_mae)
print("SimpleRNN 决定系数（R²）：", simple_rnn_r2)

# 可视化结果
labels = ['MLP', 'GRU', 'SimpleRNN']
mse_values = [mlp_mse, gru_mse, simple_rnn_mse]
mae_values = [mlp_mae, gru_mae, simple_rnn_mae]
r2_values = [mlp_r2, gru_r2, simple_rnn_r2]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, mse_values, width, label='MSE')
rects2 = ax.bar(x, mae_values, width, label='MAE')
rects3 = ax.bar(x + width, r2_values, width, label='R²')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Evaluation Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()
