#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 13:32:38 2021

@author: mutt
"""

from user_agent import generate_user_agent, generate_navigator
from pprint import pprint



book_name = "result.txt"
 
path = "/Users/mutt/Documents/GitHub/小说爬虫下载/伏天氏"


import os
#获取目标文件夹的路径
filedir = path
#获取当前文件夹中的文件名称列表  
filenames=os.listdir(filedir)

print(filenames)

list1 = [int(name.split(" ",1)[0][1:-1]) for name in filenames]
list2 = [name.split(" ",1)[1] for name in filenames]
list1,list2 = (list(t) for t in zip(*sorted(zip(list1, list2))))
list1 = ["第%s章"%name for name in list1]
list3 = list(zip(list1,list2))
list4 = [name[0]+ " " +name[1] for name in list3]

for f_path in list4:
    f_path = path + "/"+ f_path
    with open(f_path,"r") as zhengwen:
        txt = zhengwen.read()
        zhengwen.close()
        with open("/Users/mutt/Documents/GitHub/小说爬虫下载" + "/" + book_name,"a") as xs:
            xs.write("\n" + txt)
            xs.close()
        
    