# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 09:44:03 2021

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 手动构造数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
# 定义前项传播（模型）
def forward(x):
    return x*w
 
# 定义损失函数
def loss(x, y):
    y_pred = forward(x)   
    #均方误差（mean-square error, MSE）
    return (y_pred - y)**2   
 
 
# 穷举法
w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):  #np.arange 均匀取数
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)
    
plt.plot(w_list,mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()    