# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 12:48:57 2021

@author: TTMU
"""


import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from DataDeal import DataDeal

##setting
title = "数据分析师招聘情况分析 2021.3.22"
basic_path = "D:\github\-Data-Analysis\数据分析\项目\数据分析师招聘状况\\"
data_path = "全国数据分析师招聘3.22.csv"
data_df = ""
pic_color = "#CEE2F7"  


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
    st.title(AP_mode)
    
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
        
        
        
    def data_analysis(): 
        analysis_df = dd.Data_cleaning()
        sel_type = st.sidebar.selectbox("分类标准",analysis_df.columns)
        city_df = analysis_df[sel_type].value_counts().to_frame().sort_values(by = sel_type,ascending=False)
        show_num = st.sidebar.slider("",2,city_df.shape[0],10,1)
        city_df = city_df.head(show_num)
        city_df = city_df.reset_index()
        city_df.columns = [sel_type,"职位数量"]
        st.plotly_chart(go_bar(city_df,city_df[sel_type],city_df["职位数量"]))
        st.plotly_chart(go_pie(city_df[sel_type], city_df["职位数量"]))
        
        
    def go_bar(df,x_label,y_label,title="分类统计柱状图",frame_pic=1000,bg_color="#CEE2F7"):
        
        fig = px.bar(df, 
                     y = y_label,
                     x = x_label,
                     text = x_label,
                     color = y_label,
                     title = title)
        
        fig.update_traces(
            texttemplate='%{y}', 
            textposition='outside')
        
        fig.update_layout(
            uniformtext_minsize=8,
            uniformtext_mode='hide',
            legend = {"x":1,"bordercolor" : "black","borderwidth": 1},
            xaxis = {"showline":True},
            yaxis = {"ticks":"outside","tickwidth":5,"showgrid":True,"gridwidth":2,"showline":True,'title':"职位个数"},
            autosize=False,
            width=frame_pic,
            height=frame_pic/16*10,
            paper_bgcolor=bg_color, # 设置整个面板的背景色
            plot_bgcolor=bg_color,  # 设置图像部份的背景色
            )

        return fig
    
    def go_pie(labels,values,hole=.3,frame_pic=1000,bg_color="#CEE2F7"):
        fig = go.Figure(data=[go.Pie(labels=labels,
                                     values=values, 
                                     hole=hole,
                                     sort = False,
                                     textinfo='label+percent',
                                     insidetextorientation='horizontal',
                                     hovertemplate = "%{label}职位数量占总体: %{percent}共:%{value}个职位",
                                     )])
        fig.update_layout(
        autosize=False,
        width=frame_pic,
        height=frame_pic/16*10,
        paper_bgcolor=bg_color, # 设置整个面板的背景色
        plot_bgcolor=bg_color,  # 设置图像部份的背景色
        )
        return fig

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