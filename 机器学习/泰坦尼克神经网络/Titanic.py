# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:05:35 2021

@author: Administrator
"""

#头文件引用

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn

#数据预处理（数据集构建）
class TitanicDataset(Dataset):
    
    def __init__(self,filepath):
        self.data = pd.read_csv(filepath)
        self.x_data,self.y_data = TitanicDataset.prepare_data(self.data)   
        #pandas DataFrame -> numpy array -> torch tensor
        self.x_data = torch.from_numpy(np.array(self.x_data,dtype=np.float32))
        self.y_data = torch.from_numpy(np.array(self.y_data.to_frame(),dtype=np.float32))
        self.len = self.data.shape[0]
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
 
    def __len__(self):
        return self.len

    @staticmethod
    def prepare_data(df,train = True):
        if train:
            y_data = df['Survived']

        #----
        x_data = df
        dummy_fields=['Pclass', 'Sex', 'Embarked']
        for each in dummy_fields:
            dummies= pd.get_dummies(x_data[each], prefix= each, drop_first=False)
            x_data = pd.concat([x_data, dummies], axis=1)
            
        x_data['Title'] = x_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        #处理称谓
        #Mr : 0
        #Miss : 1
        #Mrs: 2
        #Others: 3
        title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                          "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                          "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }   
        x_data['Title'] = x_data['Title'].map(title_mapping)
        x_data["Age"].fillna(x_data.groupby("Title")["Age"].transform("median"), inplace=True)
        
        to_normalize=['Age','Fare']
        for each in to_normalize:
            mean, std= x_data[each].mean(), x_data[each].std()
            x_data.loc[:, each]=(x_data[each]-mean)/std
       
        fields_to_drop=['Survived','PassengerId', 'Cabin', 'Pclass', 'Name', 'Sex', 'Ticket', 'Embarked','Title']
        x_data = x_data.drop(fields_to_drop,axis=1)
        
        print(x_data)
        if train:
            return x_data , y_data
        else:
            return x_data
        


#设计模型
dataset = TitanicDataset('train.csv')
#--batch_size batch数据个数 shuffle 是否打乱数据 num_workers 多线程
train_loader = DataLoader(dataset=dataset, batch_size=100, shuffle=True, num_workers=2) 

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.linearlinear =  nn.Sequential(nn.Linear(12,6),
                     nn.ReLU(),
                     nn.Linear(6,1),              
                     nn.Sigmoid())
                
 
    def forward(self, x):
        
        x = self.linearlinear(x)
        
        return x
 
 
model = Model()

#构造损失和优化器
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

#训练循环
if __name__ == '__main__':
    
    

    for epoch in range(200):
        for i, data in enumerate(train_loader, 0): # train_loader 是先shuffle后mini_batch
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            
            print(epoch, i, loss.item())
            # for name, parms in model.named_parameters():	
            #     print('-->name:',name,'-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad)

            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()

    test = pd.read_csv('train.csv')
    test = TitanicDataset.prepare_data(test,train = False)
    test = torch.from_numpy(np.array(test,dtype=np.float32))
    
    #print(model(test))






#数据预测


    