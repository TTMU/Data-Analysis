# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:47:17 2021

@author: Administrator
"""

import pandas as pd
from pprint import pprint
import os


def list_dir_files(root_dir, ext=None):
    """
    列出文件夹中的文件, 深度遍历
    :param root_dir: 根目录
    :param ext: 后缀名
    :return: [文件路径列表, 文件名称列表]
    """
    names_list = []
    paths_list = []
    for parent, _, fileNames in os.walk(root_dir):
        for name in fileNames:
            if name.startswith('.'):  # 去除隐藏文件
                continue
            if ext:  # 根据后缀名搜索
                if name.endswith(tuple(ext)):
                    names_list.append(name)
                    paths_list.append(os.path.join(parent, name))
            else:
                names_list.append(name)
                paths_list.append(os.path.join(parent, name))
    #paths_list, names_list = sort_two_list(paths_list, names_list)
    #return paths_list, names_list
    return paths_list,names_list

#1.读取数据

Raw_data_df = pd.DataFrame(columns = ['用户ID', '商品ID','商品类目ID','行为类型','时间戳'])

for path in list_dir_files(r'D:\github\数据集\淘宝用户行为数据分析',"csv")[0]:
    try:
        df = pd.read_csv(path, names=['用户ID', '商品ID','商品类目ID','行为类型','时间戳'])
        df = df.loc[(df['时间戳'] > 1511539200 ) & (df['时间戳'] <= 1512316799)]       
        df = df.sample(frac=0.1, replace=True, random_state=1)
        Raw_data_df = Raw_data_df.append(df)
    except :
        print("load file error")
    
print(Raw_data_df)


# data_path = r'D:\github\数据集\淘宝用户行为数据分析\UserBehavior_1.csv'

# try:
#     Raw_data_df = pd.read_csv(data_path, names=['用户ID', '商品ID','商品类目ID','行为类型','时间戳'])
# except :
#     print("load file error")



# #2.描述性分析


# #时间戳转化为时间
# Raw_data_df['时间'] = pd.to_datetime(Raw_data_df['时间戳'], unit='s')
# #预览数据
# pprint(Raw_data_df.head(20))
# #查看是否有缺失值
# print(Raw_data_df.isnull().sum())
# #查看用户行为是否符合规定：即只有PV，buy，cart，fav 四类
# print(Raw_data_df['行为类型'].unique())
# #查看原始数据时间跨度
# print(Raw_data_df['时间'].min(),Raw_data_df['时间'].max())

# #3.数据清洗
# #选择2017-11-25到2017-12-03之间的数据
# Raw_data_df=Raw_data_df.loc[(Raw_data_df['时间'] > '2017-11-25') & (Raw_data_df['时间'] <= '2017-12-03')]
# #print(Raw_data_df)


# #4.数据分析

# #.用户行为概览
# user_behav_df = Raw_data_df.groupby(['行为类型'])["用户ID"].nunique()
# print(user_behav_df)
# #行为转化率漏斗数据
# count = Raw_data_df['行为类型'].value_counts()
# print(count)
# #组合展示用表
# fin_df = pd.merge(user_behav_df,count,how= 'inner',left_index=True,right_index=True)
# print(fin_df)
