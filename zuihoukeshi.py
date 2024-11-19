import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 定义每种融合方法的评估指标
weighted_avg_results = {
    'MSE': 0.10094028780655158,
    'MAE': 0.23313975041285645,
    'R²': 0.8945075148943779
}

voting_results = {
    'MSE': 0.10622260778370698,
    'MAE': 0.2346530630823869,
    'R²': 0.888986973259098
}

hybrid_results = {
    'MSE': 0.10568539349096204,
    'MAE': 0.23441884191432888,
    'R²': 0.8895484148005026
}

# 将结果存储在一个 DataFrame 中
results_df = pd.DataFrame({
    'Method': ['Weighted Average', 'Voting', 'Hybrid'],
    'MSE': [weighted_avg_results['MSE'], voting_results['MSE'], hybrid_results['MSE']],
    'MAE': [weighted_avg_results['MAE'], voting_results['MAE'], hybrid_results['MAE']],
    'R²': [weighted_avg_results['R²'], voting_results['R²'], hybrid_results['R²']]
})

# 设置绘图风格
sns.set(style="whitegrid")

# 绘制 MSE、MAE 和 R² 的柱状图
plt.figure(figsize=(14, 6))

# MSE
plt.subplot(1, 3, 1)
sns.barplot(x='Method', y='MSE', data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')
plt.ylim(min(results_df['MSE']) - 0.01, max(results_df['MSE']) + 0.01)

# MAE
plt.subplot(1, 3, 2)
sns.barplot(x='Method', y='MAE', data=results_df, palette='plasma')
plt.title('Mean Absolute Error (MAE)')
plt.ylim(min(results_df['MAE']) - 0.01, max(results_df['MAE']) + 0.01)

# R²
plt.subplot(1, 3, 3)
sns.barplot(x='Method', y='R²', data=results_df, palette='magma')
plt.title('R² Score')
plt.ylim(min(results_df['R²']) - 0.01, max(results_df['R²']) + 0.01)

plt.tight_layout()
plt.show()

# 绘制加权平均融合中各基模型的权重分布
weights = [0.4444444444444445, 0.17171717171717174, 0.3535353535353536, 0.030303030303030276]
model_names = ['MLP', 'GRU', 'RF', 'XGB']

plt.figure(figsize=(8, 6))
sns.barplot(x=model_names, y=weights, palette='coolwarm')
plt.title('Weights of Base Models in Weighted Average Fusion')
plt.ylabel('Weight')
plt.ylim(0, max(weights) + 0.05)
plt.show()






