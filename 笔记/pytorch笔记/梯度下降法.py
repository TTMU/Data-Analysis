# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 09:47:40 2021

@author: Administrator
"""
'''
注意鞍点的问题，包括马鞍形状，从不同角度看， 同一个点可能是最大值，也可能是最小值，另一个是倒数为0的点，会使梯度保持不变，权重无法更新

大致解决方式：

1）利用Hessian矩阵，判断是否为鞍点。因为Hessian在鞍点具有正负特征值，而在局部最小值点正定。

2）随机梯度，相当于给正确的梯度加了一点noise，一定程度上避免了鞍点（但是只是一定程度）,达到类似于如下公式的效果   ：

3）随机初始化起点，也有助于逃离鞍点

4）增加偶尔的随机扰动

'''




import matplotlib.pyplot as plt
 


# 数据准备
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
# 初始化权重w 
w = 1.0
 
# 定义线性模型 y = w*x
def forward(x):
    return x*w
 
# 定义损失函数  此处为MSE
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs,ys):
        y_pred = forward(x)
        cost += (y_pred - y)**2
    return cost / len(xs)
 
# 定义梯度函数   对权重w求偏导
def gradient(xs,ys):
    grad = 0
    for x, y in zip(xs,ys):
        grad += 2*x*(x*w - y)
    return grad / len(xs)
 
epoch_list = []
cost_list = []
print('predict (before training)', 4, forward(4))

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w-= 0.01 * grad_val  # 0.01 学习率
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)
 
print('predict (after training)', 4, forward(4))
plt.plot(epoch_list,cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show() 