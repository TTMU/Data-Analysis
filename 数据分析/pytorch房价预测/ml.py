
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn import  preprocessing,linear_model, svm, gaussian_process
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

train_df = pd.read_csv(r'D:\github\pytorch房价预测\train.csv')
test_df = pd.read_csv(r'D:\github\pytorch房价预测\test.csv')
out_path = r'D:\github\pytorch房价预测\ml_submission.csv'

## saleprice 分析

print(train_df['SalePrice'].describe())
#结果显示价格数据无缺失，初步显示数据波动无较大问题。

#绘制价格分布图查看数据分布
sns.displot(train_df['SalePrice'],kde=True)
plt.show()
print("偏度: %f" % train_df['SalePrice'].skew())
print("峰度: %f" % train_df['SalePrice'].kurt())
#数据呈现右偏分布，高峰处略陡

## 数据分析
#绘制相关性矩阵
#对离散数据进行处理
nonum_features = list(train_df.dtypes[train_df.dtypes == 'object'].index)
for feature in nonum_features:
    label = preprocessing.LabelEncoder()
    train_df[feature] = label.fit_transform(train_df[feature])

#相关性矩阵分析
corrmat = train_df.corr()
#相关因素
a = corrmat[corrmat['SalePrice'] > 0.3 ].index
#较强相关因素
b = corrmat[corrmat['SalePrice'] > 0.5 ].index
print(a , b )
#绘制相关热图
mask = (corrmat >-0.3) & (corrmat <0.3)
f, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(corrmat, square=True,linewidths=.5, cmap='YlGnBu',vmin = -1,vmax = 1,mask = mask )
plt.show()
'''
对相关性分析，与价格因素相关的有:
       'LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
       'Foundation', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt',
       'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF'
其中,相关性较强因素():
       'OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',
       'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea'

'''
# sc_df = train_df[b].copy()
# sns.pairplot(sc_df)
# plt.show()

b = list(b)
b.remove('SalePrice')
x = train_df[b].values
y = train_df['SalePrice'].values.reshape(-1,1)

#训练数据和测试数据的数据归一化与反归一化
'''
笔记：记录实际操作中容易出现的问题
！！！ 关于归一化和反归一化的数据统一的问题
训练集和测试集必须使用同一参数的归一化和反归一化
第一点： fit_transform() 和 transform()的区别。
两者都是归一化函数，但是fit_transform() 会储存归一化函数是的相关参数，因此对训练集使用fit_transform() ，储存了训练集归一化的相关参数，然后利用这些参数对测试集进行统一的归一化transform()
【切记不能再使用fit_transform() ，第二次使用fit_transform() 会刷新mm里储存的参数！！】 。
第二点： 反归一化时任然要使用归一化时储存的参数和格式。归一化时使用的是ss = StandardScaler()，因此后面仍然要使用mm进行反归一化；
！！！训练集、测试集、归一化、反归一化 均要使用同一参数的归一化函数！！
'''
xss = preprocessing.StandardScaler()
yss = preprocessing.StandardScaler()
#训练数据准备，交叉验证划分训练集和测试集
x_scaled = xss.fit_transform(x)   #注意这里是fit_transform
y_scaled = yss.fit_transform(y)
X_train,X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=30)
#预测数据准备
#预测数据离散值转换
test_nonum_features = list(test_df.dtypes[test_df.dtypes == 'object'].index)
for feature in test_nonum_features:
    label = preprocessing.LabelEncoder()
    test_df[feature] = label.fit_transform(test_df[feature])
print(test_df[b].isnull().sum())
test = test_df[b].values
test = xss.transform(test)   #注意这里是transform
test[np.isnan(test)] = 0

print('*' * 30)
print('模型选择')
print(train_df[b].isnull().sum())
# clfs = {
#         'svm':svm.SVR(), 
#         'RandomForestRegressor':RandomForestRegressor(n_estimators=400),
#         'BayesianRidge':linear_model.BayesianRidge()
#        }

# for clf in clfs:
#     try:
#         clfs[clf].fit(X_train, y_train.ravel())
#         y_pred = clfs[clf].predict(X_test)
#         print(clf + " cost:" + str(np.sum(y_pred-y_test)/len(y_pred)) )
#     except Exception as e:
#         print(clf + " Error:")
#         print(str(e))
print('*' * 30)
#选择随机森林算法
rfg = RandomForestRegressor(n_estimators= 400)

print(X_train.shape,y_train.ravel().shape)
print(X_train.shape,test.shape)

rfg.fit(X_train,y_train.ravel())

y_pred = rfg.predict(test)
y_pred = yss.inverse_transform(y_pred)

result = pd.DataFrame(y_pred , columns=['SalePrice'])
result['Id'] = np.array(result.index)
result['Id'] = result['Id'] + 1461


print(result)

try:
    result.to_csv(out_path,columns=['Id', 'SalePrice'],index=False)
except :
    print('fail')