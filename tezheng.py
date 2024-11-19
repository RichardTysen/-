import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, RepeatedKFold
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.svm import SVR
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
sns.set(style='ticks')

# 读取训练数据和测试数据
train_data_file = r"C:/Users/17869/Desktop/zhengqi_train.txt"
test_data_file = r"C:/Users/17869/Desktop/zhengqi_test.txt"
dtrain = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
dtest = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')
# 合并训练数据和测试数据（假设测试数据也有target列，如果没有请根据实际情况处理）
dfull = pd.concat([dtrain, dtest], ignore_index=True,sort=False)
#print('训练集大小: ', np.shape(dtrain))
#print('测试集大小: ', np.shape(dtest))

#plt.figure(figsize=(18,8),dpi=100)
#dfull.boxplot(sym='r^', patch_artist=True, notch=True)
#plt.title('DATA-FULL')
#plt.show()      箱型图

def heatmap(df):
    plt.figure(figsize=(30,22), dpi=50)
    cols = df.columns.tolist()
    mcorr = df[cols].corr(method = 'spearman')
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.zeros_like(mcorr, dtype = np.bool_)
    mask[np.triu_indices_from(mask)] = True              # 角分线右侧为True
    g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
    plt.xticks(rotation=45)
    return mcorr
dtrain_mcorr = heatmap(dtrain)
#plt.show()     热力图

# 删除dfull表中指定的列  创建特征剔除函数，方便对数据进行统一处理
def drop_var(var_lst):
    dfull.drop(var_lst, axis=1, inplace=True)
# 将dfull重新分割为dtrain核dtest   
def split_dfull():
    dtrain = dfull[dfull['target'].notnull()]
    dtest = dfull[dfull['target'].isnull()]
    dtest.drop('target', axis=1, inplace=True)
    return dtrain, dtest

# 删除相关性较低的特征
drop_var(['V5','V9','V11','V17','V20','V21','V22','V27','V28'])
dtrain, dtest = split_dfull()

# 绘制特征与target的关系
plt.figure(figsize=(20, 60), dpi=60)
# 使用enumerate简化计数
for i, col in enumerate(dtrain.columns):
    # 绘制回归图
    plt.subplot(15, 4, 2 * i + 1)
    sns.regplot(
        data=dtrain,
        x=col,
        y='target',
        scatter_kws={'marker': '.', 's': 5, 'alpha':.6},
        line_kws={'color': 'k'}
    )
    plt.title(f'{i}')
    plt.xlabel(col)
    plt.ylabel('target')

    # 绘制分布直方图
    plt.subplot(15, 4, 2 * i + 2)
    sns.distplot(dtrain[col], fit=stats.norm)
    plt.title(f'{i}')
    plt.xlabel(col)

plt.tight_layout()
#plt.show()

# 删除相关性较低的特征2
drop_var(['V14','V25','V26','V32','V33','V34','V35'])
dtrain, dtest = split_dfull()


# 分布呈明显左偏的特征
piantai = ['V0','V1','V6','V7','V8','V12','V16','V31']
# 创建函数——找到令偏态系数绝对值最小的对数转换的底
def find_min_skew(data):
    subs = list(np.arange(1.01,2,0.01))
    skews = []
    for x in subs:
        skew = abs(stats.skew(np.power(x,data)))
        skews.append(skew)
    min_skew = min(skews)
    i = skews.index(min_skew)
    return subs[i], min_skew

# 绘制偏态特征与结果变量相关图与正态分布图
plt.figure(figsize=(20,36), dpi=80)
i = 0
for col in piantai:
    # 正态化之前的偏态特征与结果变量相关图与正态分布图
    i += 1
    plt.subplot(8,4,i)
    sns.regplot(data=dtrain, x=col, y='target',
               scatter_kws = {'marker': '.', 's': 5, 'alpha':.6},
               line_kws = {'color': 'k'})
    coef = np.corrcoef(dtrain[col], dtrain['target'])[0][1]
    plt.title(f'coef = {coef}')
    plt.xlabel(col)
    plt.ylabel('target')
    
    i += 1
    plt.subplot(8,4,i)
    sns.distplot(dtrain[col], fit = stats.norm)
    plt.title(f'skew = {stats.skew(dtrain[col])}')
    plt.xlabel(col)
    
    # 找到合适的底数，对dtrain的偏态特征进行对数转换
    sub = find_min_skew(dfull[col])[0]
    dtrain[col] = np.power(sub, dtrain[col])
    
    # 对数转换之后特征与结果变量相关图与正态分布图
    i += 1
    plt.subplot(8,4,i)
    sns.regplot(data=dtrain, x=col, y='target',
               scatter_kws = {'marker': '.', 's': 5, 'alpha':.6},
               line_kws = {'color': 'k'})
    coef = np.corrcoef(dtrain[col], dtrain['target'])[0][1]
    plt.title(f'coef = {coef}')
    plt.xlabel(col+' transformed')
    plt.ylabel('target')
    
    i += 1
    plt.subplot(8,4,i)
    sns.distplot(dtrain[col], fit = stats.norm)
    plt.title(f'skew = {stats.skew(dtrain[col])}')
    plt.xlabel(col+' transformed')
    
plt.tight_layout()
#plt.show()

# 对dfull的偏态特征进行对数转换
for col in piantai:
    sub = find_min_skew(dfull[col])[0]
    dfull[col] = np.power(sub, dfull[col])
dtrain, dtest = split_dfull()

# 对dfull的特征进行标准化
dfull.iloc[:,:-1] = dfull.iloc[:,:-1].apply(lambda x: (x-x.mean())/x.std())
dtrain, dtest = split_dfull()

#查看完成后特征分布直方图
plt.figure(figsize=(20,20),dpi=80)
for i in range(22):
    plt.subplot(6,4,i+1)
    sns.distplot(dtrain.iloc[:,i], color='green')
    sns.distplot(dtest.iloc[:,i], color='red')
    plt.legend(['Train','Test'])
plt.tight_layout()
#plt.show()

#print(dtest.shape)

# 对dtrain进行PCA处理，保留16个主要成分
pca = PCA(n_components=22)
dtrain_pca = pca.fit_transform(dtrain.iloc[:,:-1])  # 对除目标列外的特征进行PCA变换
dtrain_pca_df = pd.DataFrame(dtrain_pca)  # 将变换后的数据转换为DataFrame
dtrain_pca_df['target'] = dtrain['target']  # 添加目标列

# 对dtest进行同样的PCA处理
dtest_pca = pca.transform(dtest)  # 注意这里直接使用之前fit的pca模型进行变换
dtest_pca_df = pd.DataFrame(dtest_pca)

print("PCA处理完成，dtrain_pca_df为处理后的训练集数据，dtest_pca_df为处理后的测试集数据。")
#---------------------------------------------------------------------------------------
# 用线性回归模型对dtrain_pca_df
X = np.array(dtrain_pca_df.iloc[:,:-1])
y = np.array(dtrain_pca_df['target'])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

# 定义评分函数
def score(y, y_pred):
	# 计算均方误差 MSE
    print('MSE = {0}'.format(mean_squared_error(y, y_pred)))
    # 计算模型决定系数 R2
    print('R2 = {0}'.format(r2_score(y, y_pred)))
    
    # 计算预测残差，找异常点
    y = pd.Series(y)
    y_pred = pd.Series(y_pred, index=y.index)
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()
    z = (resid - mean_resid) / std_resid
    n_outliers = sum(abs(z)>3)
    
    # 图一：真实值vs预计值
    plt.figure(figsize=(18,5), dpi=80)
    plt.subplot(131)
    plt.plot(y, y_pred, '.')
    plt.xlabel('y')
    plt.ylabel('y_pred')
    plt.title('corr = {:.3f}'.format(np.corrcoef(y,y_pred)[0][1]))
    plt.show()
    # 图二：残差分布散点图
    plt.subplot(132)
    plt.plot(y, y-y_pred, '.')
    plt.xlabel('y')
    plt.ylabel('resid')
    plt.ylim([-3,3])
    plt.title('std_resid = {:.3f}'.format(std_resid))
    plt.show()
    # 图三：残差z得分直方图
    plt.subplot(133)
    sns.distplot(z, bins=50)
    plt.xlabel('z')
    plt.title('{:.0f} samples with z>3'.format(n_outliers))
    plt.tight_layout()
    plt.show()

# 利用RidgeCV函数自动寻找最优参数
ridge = RidgeCV()
ridge.fit(X_train, y_train)
print('best_alpha = {0}'.format(ridge.alpha_))
 
y_pred = ridge.predict(X_train)
score(y_train, y_pred)


# 利用LassoCV函数自动寻找最优参数
resid = y_train - y_pred
resid = pd.Series(resid, index=range(len(y_train)))
resid_z = (resid-resid.mean()) / resid.std()
outliers = resid_z[abs(resid_z)>3].index
print(f'{len(outliers)} Outliers:')
print(outliers.tolist())
 
plt.figure(figsize=(14,6),dpi=60)

plt.subplot(121)
plt.plot(y_train, y_pred, '.')
plt.plot(y_train[outliers], y_pred[outliers], 'ro')
plt.title(f'MSE = {mean_squared_error(y_train,y_pred)}')
plt.legend(['Accepted', 'Outliers'])
plt.xlabel('y_train')
plt.ylabel('y_pred')
 
plt.subplot(122)
sns.distplot(resid_z, bins = 50)
sns.distplot(resid_z.loc[outliers], bins = 50, color = 'r')
plt.legend(['Accepted', 'Outliers'])
plt.xlabel('z')
plt.tight_layout()
#plt.show()
# 异常样本点剔除
X_train = np.array(pd.DataFrame(X_train).drop(outliers,axis=0))
y_train = np.array(pd.Series(y_train).drop(outliers,axis=0))


# 将dtrain_pca_df保存为txt文件，格式与初始训练集一致（假设初始训练集是以制表符分隔的）
dtrain_pca_df.to_csv(r"C:/Users/17869/Desktop/zhengqi_train_processed.txt", sep='\t', encoding='utf-8', index=False)
# 将dtest_pca_df保存为txt文件，格式与初始测试集一致（假设初始测试集是以制表符分隔的）
dtest_pca_df.to_csv(r"C:/Users/17869/Desktop/zhengqi_test_processed.txt", sep='\t', encoding='utf-8', index=False)