#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 13:53:11 2021

@author: mutt
"""
import os

#web展示
import streamlit as st

#数据处理
import numpy as np
import pandas as pd

#机器学习


#可视化
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns



#路径处理错误
fold_path = os.path.abspath(os.path.dirname(__file__)) + "/"

#  数据预处理
#  |----------------读取数据
try :
    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')
except :
    
    train = pd.read_csv(fold_path + 'input/train.csv')
    test = pd.read_csv(fold_path + 'input/test.csv')


#  |----------------特征工程
train_default = train.copy()
test_default = test.copy()

train_test_data = [train, test] # combining train and test dataset

#正则提取名字前面 xx.  作为title列
for dataset in train_test_data:
        dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)         


#---------------------------------------ui展示用数据
train_f = train.copy()   #-
test_f = test.copy()    #-
#---------------------------------------


#处理称谓
#Mr : 0
#Miss : 1
#Mrs: 2
#Others: 3
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


#处理性别
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    

#删除不必要数据

train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
    
#年龄缺失  用title对应的age中位数填充 (Mr, Mrs, Miss, Others)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

#---------------------------------------ui展示用数据
train_age = train.copy()   #-
test_age = test.copy()
#---------------------------------------

#年龄分类 child: 0 young: 1 adult: 2 mid-age: 3 senior: 4  
def change_age(age):
    if age <= 16:
        return 0
    elif age >16 and age <=26:
        return 1
    elif age >26 and age <=36:
        return 2
    elif age >36 and age <=62:
        return 3
    else:
        return 4
for dataset in train_test_data:
    dataset["Age"] = dataset["Age"].apply(lambda x:change_age(x))

#填补缺失值   123等级舱大多是s港口上船的  故用s填充所有港口
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
Pclass_df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
Pclass_df.index = ['1st class','2nd class', '3rd class']

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
#更新数据
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

#填补船票缺失值  根据船舱等级对应的船票中位数填充
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

#---------------------------------------ui展示用数据
train_fare = train.copy()   #-
test_fare = test.copy()
#---------------------------------------

def change_fare(fare):
    if fare <= 17:
        return 0
    elif fare >17 and fare <=30:
        return 1
    elif fare >30 and fare <=100:
        return 2
    else:
        return 3
for dataset in train_test_data:
    dataset["Fare"] = dataset["Fare"].apply(lambda x:change_fare(x))
    
#船舱数据处理
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
#填补空缺值
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

#家庭规模
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
train_fam = train.copy()   

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)

#丢弃不需要的数据
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)

train_data = train.drop('Survived', axis=1)
target = train['Survived']



#建模










 
    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#交叉验证
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
   
#kNN 
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
#knn分数  
round(np.mean(score)*100, 2)

#决策树
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
#决策树分数
round(np.mean(score)*100, 2)
    
#随机森林
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
#随机森林分数
round(np.mean(score)*100, 2)

#朴素贝叶斯
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
#朴素贝叶斯分数

#svm支持向量机
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
#svm分数
round(np.mean(score)*100,2)



    
#测试
clf = SVC()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })


submission.to_csv('submission.csv', index=False)


















#ui 

#fun
def data_shape (df,title ):
    data_shape_str = title + "共" + str(df.shape[0]) + "行，共" + str(df.shape[1]) + "列"
    return data_shape_str

def sur_bar_chart(df , feature , mode = "group" ,key = "",pic_title =""):     # mode( group ,  stack)
    
    survived = df[df['Survived']==1][feature].value_counts()
    dead = df[df['Survived']==0][feature].value_counts()
 
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    
    fig = go.Figure()
    for i in range(df.shape[1]):
        fig.add_trace(
            go.Bar(name=str(df.columns[i]), x=df.index, y= df[df.columns[i]])
            )

    # Change the bar mode
    key_str = key + "柱状分组样式"
    if st.sidebar.checkbox(key_str):
        fig.update_layout(barmode="group")
    else :
        fig.update_layout(barmode="stack")
    
    #fig.update_layout(barmode="stack")
        
    fig.update_traces(texttemplate='%{y}人', 
                      textposition="auto",
                      )
    fig.update_layout(
                            legend = {"x":1,"bordercolor" : "black","borderwidth": 1},
                            xaxis = {"showline":True},
                            yaxis = {"ticks":"outside","tickwidth":5,"showgrid":True,"gridwidth":2,"showline":True},
                            title = pic_title
                            )
        
    
    return fig
    
    
    
def pie_chart():
    pass
    


#     streamlit 设置
app_title = "泰坦尼克生存率预测"
st.set_page_config(page_title=app_title,layout="wide",page_icon = ":passenger_ship:") 
st.sidebar.title(app_title)
st.info("Age(年龄)、Cabin(船舱号)、Embarked(登船港口)、Fare(船票价格)、Name(姓名)、Parch(不同代直系亲属)、PassengerId(乘客编号)、Pclass(船舱等级)、Sex(性别)、SibSp(同代直系亲属)、survived(生存情况)、Ticket(船票编码)")

with st.beta_expander("原始数据"):  
    st.subheader("训练集")
    st.dataframe(train_default)
    st.subheader("测试集")
    st.dataframe(train_default)

with st.beta_expander("数据形状"):  
    train_info, test_info = st.beta_columns(2)
    
    with train_info:
        st.subheader("训练集")
        st.write(data_shape(train_default,"1.训练集"))
        st.write("2.空值统计")
        st.table(train_default.isnull().sum())
        
    with test_info :
        st.subheader("测试集")
        st.write(data_shape(test_default,"1.测试集"))
        st.write("2.空值统计")
        st.table(test_default.isnull().sum())
    st.markdown("** 分析 ： **")
    st.write("1.训练集有177行年龄信息缺失，687行船舱信息缺失，2行登船信息缺失")
    st.write("2.测试集有86行年龄信息缺失，327行船舱信息缺失，无登船信息")
  
st.sidebar.subheader("设置")
with st.beta_expander("训练集分类统计"):  
    sel_list = ["Sex","Pclass","SibSp","Parch","Embarked",]
    sel_pil = st.selectbox("分类",sel_list )
    st.plotly_chart(sur_bar_chart(train_default,sel_pil,key = "训练集分类统计"))
    
with st.beta_expander("特征工程"):  
    st.subheader("提取 XX. 作Title列 eg: Mrs. ")
    st.write(train_f)
    st.write(test_f)
    
    st.subheader("Title 统计")
    train_info, test_info = st.beta_columns(2)  
    
    with train_info:
        st.write("训练")
        st.write(train_f['Title'].value_counts())
        st.write(train_test_data[0]['Title'].value_counts())
    
    with test_info:
        st.write("测试")
        st.write(test_f['Title'].value_counts())
        st.write(train_test_data[1]['Title'].value_counts())
    st.plotly_chart(sur_bar_chart(train_test_data[0],"Title",key = "特征工程title",pic_title= "Name >>Mr : 0|Miss : 1|Mrs: 2|Others: 3"))
    st.plotly_chart(sur_bar_chart(train,"Sex",key = "特征工程sex",pic_title= "Sex >>male : 0 | female : 1"))
    

    st.plotly_chart(sur_bar_chart(train,"Age",key = "特征工程Age",pic_title= "年龄分类 child: 0 young: 1 adult: 2 mid-age: 3 senior: 4"))
    

with st.beta_expander("填补缺失值结果"): 
    st.subheader("年龄生存分布")
    facet = sns.FacetGrid(train_age, hue="Survived",aspect=4)
    facet.map(sns.kdeplot,'Age',shade= True)
    facet.set(xlim=(0, train_age['Age'].max()))
    facet.add_legend()
    st.pyplot(facet)
    
    st.subheader("船票生存分布")
    facet = sns.FacetGrid(train, hue="Survived",aspect=4)
    facet.map(sns.kdeplot,'Fare',shade= True)
    facet.set(xlim=(0, train['Fare'].max()))
    facet.add_legend()
    st.pyplot(facet)
    
    st.subheader("家庭规模生存分布")
    facet = sns.FacetGrid(train_fam, hue="Survived",aspect=4)
    facet.map(sns.kdeplot,'FamilySize',shade= True)
    facet.set(xlim=(0, train_fam['FamilySize'].max()))
    facet.add_legend()
    st.pyplot(facet)
    
    st.write(train)
    

with st.beta_expander("测试"):
    pre_info, real_info = st.beta_columns(2)
    with pre_info:
        st.write(submission)
    with real_info:
        st.write(test)


    
    
    
    

    
    
    
    
    
