
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class Deal_data():
    def __init__(self,path,train = True,y_label_name = None):
        self.data = pd.read_csv(path)
        self.data_fillna()
        self.data = self.data.drop('Id',axis = 1)
        self.istrain = train
        self.y_label_name = y_label_name 
        if self.istrain:
            self.y_labels = self.data[y_label_name]
            self.data = self.data.drop(y_label_name,axis = 1)

    def df_num_data(self):
        num_features = list(self.data.dtypes[self.data.dtypes != 'object'].index)
        return self.data[num_features],num_features
        
    def df_nonum_data(self):
        _ , num_data = self.df_num_data()
        non_num_columns = [col for col in list(self.data.columns) if col not in num_data]
        return self.data[non_num_columns] , non_num_columns

    def get_y_labels(self):
        if self.istrain:
            return self.y_labels
        else:
            print("测试数据无labels值")
    
    def data_fillna(self):
        #连续数值填补空值
        self.data[self.df_num_data()[1]] = self.data[self.df_num_data()[1]].fillna(0)
        #离散值填补空值
        self.data[self.df_nonum_data()[1]] = self.data[self.df_nonum_data()[1]].fillna('NaN')

    def change_nonum2num(self):
        non_num_columns = self.df_nonum_data()[1]
        nonnum2num_df = self.data[self.df_nonum_data()[1]].copy()
        for col in non_num_columns:
            unique_value = nonnum2num_df[col].unique()
            for idx,value in enumerate(unique_value):
                nonnum2num_df[col] = nonnum2num_df[col].replace(value,idx + 1)
        return nonnum2num_df

    def Entire_data(self,minmax = False,all = False):
        if minmax == True:
            num_df = Deal_data.Mean_normalization(self.df_num_data()[0])
            nonum_df = self.change_nonum2num()
            nonum_df = Deal_data.Mean_normalization(nonum_df)
            entire_df = pd.concat([num_df,nonum_df],axis=1)
        else:
            num_df = self.df_num_data()[0]
            nonum_df = self.change_nonum2num()
            entire_df = pd.concat([num_df,nonum_df],axis=1)

        if all:
            entire_df[self.y_label_name] = self.y_labels
        return entire_df

    def Mean_normalization(df):       
        return (df - df.mean())/(df.max() - df.min())

    def z_score(df):
        return (df - df.mean())/df.std()



    

#建立神经网络模型
class Net(nn.Module):
    def __init__(self, Data_in, M1, M2, M3, Data_out):
        super(Net, self).__init__()
        
        self.linear1 = nn.Linear(Data_in, M1)
        self.linear2 = nn.Linear(M1, M2)
        self.linear3 = nn.Linear(M2, M3)
        self.linear4 = nn.Linear(M3, Data_out)
        self.relu = nn.ReLU()

        #self.line1 = nn.Linear(Data_in,1)
        
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        return self.linear4(x)

        #return self.line1(x)
    
#对数均方根误差

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


#--------------model 1---------
def train1(Data_in,M1,M2,M3,Data_out,train_num,labels):
    model1 = Net(Data_in,M1,M2,M3,Data_out)
    model1.to(device)

    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model1.parameters(), lr=1e-4)

    losses1 = []
    print(train_num)
    for t in range(500):
        y_pred = model1(train_num)
        
        loss = criterion(y_pred, labels)
        print('model  >>  1  << loss:',t, loss.item())
        losses1.append(loss.item())
        
        if torch.isnan(loss):
            break
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return losses1



#--------------model 2---------

def train2(Data_in,M1,M2,M3,Data_out,train_num,labels):
    losses2 = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model2 = Net(Data_in,M1,M2,M3,Data_out)

    model2.to(device)

    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model2.parameters(), lr=1e-4)

    for t in range(500):
        

        y_pred = model2(train_num)
        loss = criterion(y_pred, labels)
        print(t, loss.item())
        losses2.append(loss.item())
        
        if torch.isnan(loss):
            break
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return losses2

#--------------model 3---------

def train3(Data_in,M1,M2,M3,Data_out,train_num,labels,lr_num = 1e-4):
    losses3 = []
    model3 = Net(Data_in,M1,M2,M3,Data_out)
    model3.cuda()
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model3.parameters(), lr=lr_num)

    for t in range(500):
        

        y_pred = model3(train_num)
        loss = criterion(y_pred, labels)
        print(t, loss.item())
        losses3.append(loss.item())
        
        if torch.isnan(loss):
            break
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return losses3

#--------------model 4---------
def train4(Data_in,M1,M2,M3,Data_out,train_num,labels,lr_num = 1e-4):
    losses4 = []
    model4 = Net(Data_in,M1,M2,M3,Data_out)
    model4.cuda()
    criterion = nn.MSELoss(reduction='sum')
    #criterion = RMSLELoss()
    optimizer = torch.optim.Adam(model4.parameters(), lr=lr_num)

    for t in range(500):
        

        y_pred = model4(train_num)
        loss = criterion(y_pred, labels)
        print(t, loss.item())
        losses4.append(loss.item())
        
        if torch.isnan(loss):
            break
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return losses4,model4

#--------------model 5---------



if __name__ == "__main__":
    #GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_df = Deal_data(r'D:\github\pytorch房价预测\train.csv',y_label_name= 'SalePrice')


    #连续数据
    train_num_df,train_num_df_cl = train_df.df_num_data()
    train_nonum_df,train_nonum_df_cl = train_df.df_nonum_data()
    labels = train_df.get_y_labels()

    #连续数据
    train_num = torch.tensor(train_num_df.values,dtype = torch.float)
    labels = torch.tensor(labels.values,dtype = torch.float)
    labels = labels.view((-1,1))

    #离散数据
    train_nonum_df = train_df.change_nonum2num()
    train_nonum_df = Deal_data.Mean_normalization(train_nonum_df)
    #train_nonum_df = Deal_data.z_score(train_nonum_df)
    train_nonum = torch.tensor(train_nonum_df.values,dtype = torch.float)

    #数据迁移到gpu上
    train_num = train_num.to(device)
    labels = labels.to(device)
    train_nonum = train_nonum.to(device)

    #定义神经网络参数
    M1 , M2 , M3 = 500,1000,200
    Data_in,Data_out = train_num.shape[1],labels.shape[1]

    #保留缩放特征
    feature = train_df.Entire_data(all = True)
    train_fea = feature.agg(['max','min','mean'])
 

    # #训练模型1   产生了梯度爆炸现象
    # # loss1 = train1(Data_in,M1,M2,M3,Data_out,train_num,labels)

    #模型2数据准备
    train_num_df = Deal_data.Mean_normalization(train_num_df)
    labels = Deal_data.Mean_normalization(train_df.get_y_labels())

    train_num = torch.tensor(train_num_df.values,dtype = torch.float)
    labels = torch.tensor(labels.values,dtype = torch.float)
    labels = labels.view((-1,1))

    train_num = train_num.to(device)
    labels = labels.to(device)

    # #训练模型2  数据处理后实现模型收敛


    # loss2 = train2(Data_in,M1,M2,M3,Data_out,train_num,labels)
    # plt.figure(figsize=(12, 10))
    # plt.plot(range(len(loss2)), loss2,label = "nor data")


    #训练模型3   探索不同学习率对收敛效果的影响
    lr_list = [0.00016,0.00018,0.00020,0.00022,0.00023]
    total_loss = []
    for lr in lr_list:
            loss = train3(Data_in,M1,M2,M3,Data_out,train_num,labels,lr)
            total_loss.append(loss)
    
    for idx, loss in enumerate(total_loss, 0):
        plt.plot(range(len(loss)), loss,label='lr:'+str(lr_list[idx]))
    
    

    #模型4  由model3 可以看出学习率在0.0002左右效果不错
    loss4 = train4(Data_in,M1,M2,M3,Data_out,train_num,labels,lr_num=0.00020)[0]
    plt.plot(range(len(loss4)), loss4,label = "num Adam lr = 0.0002")

    

    #模型5  使用模型4，但是传入数据为train_nonum
    Data_in,Data_out = train_nonum.shape[1],labels.shape[1]
    loss5 = train4(Data_in,M1,M2,M3,Data_out,train_nonum,labels,lr_num=0.00020)[0]
    plt.plot(range(len(loss5)), loss5,label = "nonnum Adam lr = 0.0002")

    #模型6  全部数据进行拟合
    Entire_df = train_df.Entire_data(minmax= True)  
    Entire_tensor = torch.tensor(Entire_df.values,dtype = torch.float)
    Entire_tensor = Entire_tensor.to(device)

    Data_in,Data_out = Entire_tensor.shape[1],labels.shape[1]
    loss6 , model = train4(Data_in,M1,M2,M3,Data_out,Entire_tensor,labels,lr_num=0.00020)
    plt.plot(range(len(loss6)), loss6,label = "Entire_data Adam lr = 0.0002")
    
    #数据展示
    plt.xlabel('model')
    plt.legend(loc='upper right')
    plt.show()

    #测试集数据准备
    print(train_fea.loc['max','SalePrice'])

    test_df = Deal_data(r'D:\github\pytorch房价预测\test.csv',train = False)
    test = test_df.Entire_data()

    for col in test.columns:
        test[col] = (test[col] - train_fea.loc['mean',col])/(train_fea.loc['max',col] - train_fea.loc['min',col])
    


    test_tensor = torch.tensor(test.values,dtype = torch.float)
    test_tensor = test_tensor.to(device)
    y_pre = model(test_tensor)
    y_pre = y_pre.cpu()



    result = pd.DataFrame(y_pre.data.numpy(), columns=['SalePrice'])
    result['SalePrice'] = result['SalePrice'].fillna(0)
    result['SalePrice'] = result['SalePrice'] * (train_fea.loc['max','SalePrice'] - train_fea.loc['min','SalePrice']) + train_fea.loc['mean','SalePrice']
    result['Id'] = np.array(result.index)
    result['Id'] = result['Id'] + 1461
    

    print(result)
    
    try:
        result.to_csv('./submission.csv',columns=['Id', 'SalePrice'],index=False)
    except :
        print('fail')
