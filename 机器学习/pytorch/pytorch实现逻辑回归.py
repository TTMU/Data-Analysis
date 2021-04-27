# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 12:55:20 2021

@author: Administrator
"""

'''
逻辑斯蒂回归和线性模型的明显区别是在线性模型的后面，添加了激活函数(非线性变换)

Pytorch中CrossEntropyLoss()函数的主要是将softmax-log-NLLLoss合并到一块得到的结果。

    1、Softmax后的数值都在0~1之间，所以ln之后值域是负无穷到0。

    2、然后将Softmax之后的结果取log，将乘法改成加法减少计算量，同时保障函数的单调性 。

    3、NLLLoss的结果就是把上面的输出与Label对应的那个值拿出来(下面例子中就是：将log_output\logsoftmax_output中与y_target对应的值拿出来)，求均值，加负号。

'''


import torch
# import torch.nn.functional as F
 
# prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])
 
#design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)
 
    def forward(self, x):
        # y_pred = F.sigmoid(self.linear(x))
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegressionModel()
 
# construct loss and optimizer
# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
criterion = torch.nn.BCELoss(size_average = False) 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
 
# training cycle forward, backward, update
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
 
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)