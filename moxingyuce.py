import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# 之前的代码部分，用于读取处理后的训练集数据
train_data_file = r"C:/Users/17869/Desktop/zhengqi_train_processed.txt"
dtrain_pca_df = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')

# 提取训练集的特征和目标变量
X_full = np.array(dtrain_pca_df.iloc[:, :-1])
y_full = np.array(dtrain_pca_df['target'])

X_full.shape, y_full.shape
# 将训练集合理划分为新的训练集和测试集，这里设置测试集占比为0.2（可根据需求调整）
X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test, cv=5, n_runs=3):
    """
    训练模型并评估其性能。
    
    :param model: 模型实例
    :param model_name: 模型名称
    :param X_train: 训练集特征
    :param y_train: 训练集标签
    :param X_test: 测试集特征
    :param y_test: 测试集标签
    :param cv: 交叉验证的折数
    :param n_runs: 运行的次数
    :return: 性能指标的平均值和标准差
    """
    mse_scores = []
    r2_scores = []

    for _ in range(n_runs):
        # 交叉验证
        mse_cv = -cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
        r2_cv = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')

        # 记录结果
        mse_scores.append(mse_cv.mean())
        r2_scores.append(r2_cv.mean())

        # 拟合模型
        model.fit(X_train, y_train)

        # 预测
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # 计算训练集和测试集上的性能指标
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)



        # 训练集残差图
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(y_train, y_train - y_train_pred, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot (Training Set)')
        plt.xlabel('True Values')
        plt.ylabel('Residuals')

        # 测试集残差图
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_test - y_test_pred, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot (Test Set)')
        plt.xlabel('True Values')
        plt.ylabel('Residuals')
        plt.tight_layout()
        plt.show()

        print(f"{model_name} - Run {_+1}:")
        print(f"  Training MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}")
        print(f"  Training R2: {r2_train:.4f}, Test R2: {r2_test:.4f}")

    # 计算平均值和标准差
    mse_mean = np.mean(mse_scores)
    mse_std = np.std(mse_scores)
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)

    return mse_mean, mse_std, r2_mean, r2_std

def plot_results(models, metrics):
    """
    绘制模型性能指标的图表。
    
    :param models: 模型名称列表
    :param metrics: 性能指标字典，包含平均值和标准差
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # MSE
    mse_means = [metrics[model]['mse_mean'] for model in models]
    mse_stds = [metrics[model]['mse_std'] for model in models]
    ax[0].bar(models, mse_means, yerr=mse_stds, capsize=10)
    ax[0].set_title('Mean Squared Error (MSE)')
    ax[0].set_ylabel('MSE')
    ax[0].set_xticklabels(models, rotation=45, ha='right')

    # R2
    r2_means = [metrics[model]['r2_mean'] for model in models]
    r2_stds = [metrics[model]['r2_std'] for model in models]
    ax[1].bar(models, r2_means, yerr=r2_stds, capsize=10)
    ax[1].set_title('R2 Score')
    ax[1].set_ylabel('R2 Score')
    ax[1].set_xticklabels(models, rotation=45, ha='right')

      

    plt.tight_layout()
    plt.show()

# 定义模型
models = {
    'LassoCV': LassoCV(cv=5, random_state=42),
    'XGB': XGBRegressor(),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# 存储性能指标
metrics = {}

# 训练和评估每个模型
for model_name, model in models.items():
    mse_mean, mse_std, r2_mean, r2_std = train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test)
    metrics[model_name] = {'mse_mean': mse_mean, 'mse_std': mse_std, 'r2_mean': r2_mean, 'r2_std': r2_std}

# 绘制结果
plot_results(list(models.keys()), metrics)

