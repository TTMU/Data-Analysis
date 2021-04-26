# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter

import cn2an


class vbiPipeline:
    def process_item(self, item, spider):
          
        path = item["xs_name_path"]
        
        zj = item["zhangjie"]
        
        try :
            
            
            
            zj = zj.split(" ")
            
            
            
            zj[0] = cn2an.transform(zj[0], "cn2an")
            
            
            zj = zj[0] + " " + zj[1]
        except:
            print("章节数字转换错误")
            return item
        
        t_url = item["zhengwen"]
        
        print("正在下载>>>", path , "章节名" , zj)
        
        path = path + "/" + zj
        
        with open(path,'w') as f:    #设置文件对象
            f.write(zj + "\n" + t_url)
            f.close()
        
        return item
    
    def close_spider(self,spider):
        
        
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
                
        
                
        print(">>>>>>>>>>>>close spider<<<<<<<<<<<<<<<")
       