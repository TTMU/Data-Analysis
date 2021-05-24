'''
Author       : ttmu
Date         : 2021-05-20 11:04:50
LastEditors  : ttmu
LastEditTime : 2021-05-20 13:28:10
Description  : RMF   k-means
'''
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
n = 3

raw_df = pd.read_csv(r"D:\github\-Data-Analysis\数据分析\化妆品数据分析\化妆品_已清洗.csv")
raw_df['订单日期']=pd.to_datetime(raw_df['订单日期'],format='%Y-%m-%d')
r_df = raw_df.groupby('客户编码')['订单日期'].max().reset_index()
dmax = raw_df['订单日期'].max()
r_df['R']=(dmax-r_df['订单日期']).map(lambda x:x.days)
r_df.drop('订单日期',axis=1,inplace=True)
f_df = raw_df.groupby('客户编码')['订单编码'].count().reset_index()
m_df = raw_df.groupby('客户编码')['金额'].sum().reset_index()

rmf_df = r_df.merge(f_df)
rmf_df = rmf_df.merge(m_df)
rmf_df = rmf_df.set_index("客户编码")
rmf_df.columns = ['R','F','M']

print(rmf_df)

z_soc = (rmf_df-rmf_df.mean())/rmf_df.std()
SSD = []
centers = []
# for n in range(1,9):
#     kmeans = KMeans(n_clusters=n, random_state=0).fit(z_soc.values)
#     SSD.append(kmeans.inertia_)
#     centers.append(kmeans.cluster_centers_)
# X = range(1,9)
# plt.plot(X,SSD,"r-")
# plt.show()


kmeans = KMeans(n_clusters = n).fit(z_soc.values)
SSD.append(kmeans.inertia_)
centers.append(kmeans.cluster_centers_)

print(centers)
print(kmeans.labels_)
rmf_df['labels'] = kmeans.labels_
print(rmf_df)

rmf_df.to_csv("D:\github\-Data-Analysis\数据分析\化妆品数据分析\化妆品_rmf.csv")