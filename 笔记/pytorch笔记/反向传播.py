# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:04:45 2021

@author: Administrator
"""

'''

1.Tensor中包含data和grad，data和grad也是Tensorl。grad初始为None，调用l.backward()方法后w.grad为Tensor，故更新w.data时需使用w.grad.data。如果w需要计算梯度，
那构建的计算图中，跟w相关的tensor都默认需要计算梯度。

2.反向传播主要体现在，l.backward()。调用该方法后w.grad由None更新为Tensor类型，且w.grad.data的值用于后续w.data的更新。

3.l.backward()会把计算图中所有需要梯度(grad)的地方都会求出来，然后把梯度都存在对应的待求的参数中，最终计算图被释放。

4.取tensor中的data是不会构建  计算图  的。  


'''

import torch
a = torch.tensor([1.0])
a.requires_grad = True # 或者 a.requires_grad_()
print(a)                    #tensor([1.], requires_grad=True)
print(a.data)               #tensor([1.])
print(a.type())             #torch.FloatTensor
print(a.data.type())        #torch.FloatTensor
print(a.grad)               #None
print(type(a.grad))         #<class 'NoneType'>

#------------------------------------------------------------------------


import torch
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
w = torch.tensor([1.0]) # 权重初值为1.0
w.requires_grad = True # 需要计算梯度
 
def forward(x):
    return x*w  # w是一个Tensor
 
 
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2   #mse损失函数
 
print("predict (before training)", 4, forward(4).item())
 
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l =loss(x,y) # l是一个张量，tensor主要是在建立计算图 forward, 计算损失
        l.backward() # 反向传播，计算requires_grad为true的梯度
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data   # 权重更新时，需要用到标量，注意grad也是一个tensor，此处0.01是学习率
 
        w.grad.data.zero_() # 权重更新后要将梯度重置
 
    print('progress:', epoch, l.item()) # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图） ！！！此处注意！！！
 
print("predict (after training)", 4, forward(4).item())