# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:48:57 2021

@author: TTMU
"""

import numpy as np
import pandas as pd
import streamlit as st

from DataDeal import DataDeal

##setting
title = "数据分析师招聘情况分析 2021.3.22"
basic_path = "D:\github\-Data-Analysis\数据分析\项目\数据分析师招聘状况\\"
data_path = "全国数据分析师招聘3.22.csv"
data_df = ""


def read_file(data_file, mode='more'):
    """
    读文件, 原文件和数据文件
    :return: 单行或数组
    """
    try:
        with open(data_file, 'r' , encoding ='UTF-8') as f:
            if mode == 'one':
                output = f.read()
                return output
            elif mode == 'more':
                output = f.readlines()
                return map(str.strip, output)
            else:
                return list()
    except IOError:
        return list()

def installUI():
    
    AP_dic = {
                "原始数据":"Raw_data",
                "需求分析":"demand_analysis",
                "数据清洗":"Data_cleaning", 
                "数据分析":"data_analysis", 
                "数据报告":"Data_report", 
                "项目总结":"Project_summary", 
                }
    AP_mode = st.sidebar.selectbox("",list(AP_dic.keys()))
    
    st.sidebar.subheader(AP_mode)  
    dd = DataDeal(basic_path + data_path)
    data_df = dd.Raw_data()
    
    def Raw_data():       
        st.write(data_df)
        pass
    
    def demand_analysis():
        st.markdown(read_file(basic_path +"需求分析.md","one")) 
        pass
    
    def Data_cleaning():

        df = data_df
        
        if st.sidebar.checkbox("数据预处理（拆分数据，处理异常值，重排序）",True):
            df = dd.Data_cleaning()

        if st.sidebar.checkbox("移除描述",True):
            df.drop(['描述'],axis=1, inplace=True)
            
        if st.sidebar.checkbox("移除网址链接",True):
            df.drop(['名称_链接','公司_链接'],axis=1, inplace=True)
        
        
        print(df.info())
        st.write("当前数据共包含:",df.shape[0],"行（rows）,",df.shape[1],"列（columns）")
        st.write("数据集索引样式：",df.index)
        
        if st.sidebar.checkbox("显示全部数据"):
            st.table(df)
        else:
            st.write(df)
        pass

    def data_analysis():
        
        pass

    def Data_report():
        st.markdown(read_file(basic_path +"数据报告.md","one")) 
        pass

    def Project_summary():
        
        pass
    
    
    
    
    
    
    
    
    
#-----------------------------------
    if AP_mode in list(AP_dic.keys()):
        eval("%s()" % (AP_dic[AP_mode]))  
#------------------------------------
    


   
def main():
    st.set_page_config(page_title=title,layout="wide",page_icon = ":passenger_ship:") 
    installUI()

if __name__ == "__main__":
    main()