# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 13:05:34 2021

@author: ttmu
"""

import pandas as pd

class DataDeal:
    
    def __init__(self,filePath):
        
        fileType = filePath.split(".")[-1] 
        if fileType == "csv":
            self.data = pd.read_csv(filePath)
        elif fileType in ("xls","xlxs"):
            self.data = pd.read_excel(filePath)
        else:
            self.data = pd.DataFrame()
            print("数据读取失败，检查数据文件类型")
        
    
    def More_setting(self):
        
        pass
    
    def Data_cleaning(self):
         
        df = self.data.copy()
        df['城市'] = df['工作地点'].apply(lambda x: x.split('·')[0])
        df['区域'] = df['工作地点'].apply(lambda x: x.split('·')[1] if len(x.split('·')) >1 else '无' )
        df['详细地址'] = df['工作地点'].apply(lambda x: x.split('·')[2] if len(x.split('·')) >2 else '无' )
        df['薪酬'] = df['薪酬'].apply(lambda x: x.split('·')[0].replace('K','') if len(x.split('·')) > 0 else '无数据' )
        df['薪酬MIN'] = df['薪酬'].apply(lambda x: x.split('-')[0] if len(x.split('-')) >0 else '无' )
        df['薪酬MAX'] = df['薪酬'].apply(lambda x: x.split('-')[1] if len(x.split('-')) >1 else '无' )
        df['薪酬期数'] = df['薪酬'].apply(lambda x: x.split('·')[1].replace('薪','') if len(x.split('·')) > 1 else '12' )
        level_list = ['大专','本科','硕士','博士']
        df["学历"] = df['工作经验'].apply(lambda x : DataDeal.which(x,level_list))
        work_list = ['1-3年','3-5年','5-10年','经验不限','应届','1年以内']
        df["工作经验"] = df['工作经验'].apply(lambda x : DataDeal.which(x,work_list))
        ctype_list = ['互联网','电子商务','计算机','信息安全','应届','游戏']
        df["公司类别"] = df['公司规模'].apply(lambda x : DataDeal.which(x,ctype_list))      
        df['规模'] = df['公司规模'].str.extract(r'([1-9]\d*-*[1-9]*\d*)', expand=False)
        order_list = ['公司名字','名称','标签','城市','区域','详细地址','薪酬MIN','薪酬MAX','薪酬期数','工作经验','学历','公司类别','规模','描述','名称_链接','公司_链接',]
        df = df[order_list]      
        return df
    
    
    def Raw_data(self):    
        
        return self.data
    
    def which(string,aim_list):
        
            for aim in aim_list:
                if aim in string:
                    return aim
            return "其他"
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                