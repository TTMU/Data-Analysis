import scrapy

from xs_spider.items import vbSpiderItem

import os






zhangjie_list = []
down_path = "名字识别错误"
root_path = "/Users/mutt/Documents/GitHub/小说爬虫下载/"


class VbiqugeSpySpider(scrapy.Spider):
    
    name = 'vbiquge_spy'
    
    allowed_domains = ['vbiquge.com']
    
    start_urls = ['https://www.vbiquge.com']
    
    def __init__(self, search_url=None, *args, **kwargs):
        
        super(VbiqugeSpySpider, self).__init__(*args, **kwargs)
        
        print("------------------>",search_url)
        
        
        self.start_urls = search_url
      
    def start_requests(self):
        
        
        print("当前抓取的 Url 是：" + self.start_urls)
        yield scrapy.Request(self.start_urls)


    
    #针对单独爬虫设置header
    
    custom_settings = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.104 Safari/537.36',
    "accept-encoding" : "gzip, deflate, br",
    " accept-language" :"en,zh-CN;q=0.9,zh;q=0.8",
    "accept" : "*/*",
    "dnt:" : "1",
    "x-requested-with" : "",
    }
    
    
    def parse(self, response):
        
        title = scrapy.Selector(response).xpath('//*[@id="info"]/h1/text()').extract()[0]
        
        global down_path 
        
        down_path = root_path + title
        
        if os.path.exists(down_path):
            pass
        else:
            os.mkdir(down_path)

        zhangjie = scrapy.Selector(response).xpath('//*[@id="list"]/dl//dd/a/text()').extract()
        
        txt_url = scrapy.Selector(response).xpath('//*[@id="list"]/dl//dd/a/@href').extract()
        
        #print(title,zhangjie,txt_url)
        
        
        
        for zj,t_url in zip(zhangjie,txt_url):
            if zj not in zhangjie_list:
                
                zhangjie_list.append(zj)
                
                t_url = 'https://www.vbiquge.com' + t_url
                
                
                yield scrapy.Request(t_url,callback =self.pic_download,dont_filter = True)
                
        
    def pic_download(self,response):
        
        text = scrapy.Selector(response).xpath('//*[@id="content"]//text()').extract()
        
        textstr = "\n".join(text)
        
        textstr = textstr.replace(u'\xa0', u' ')
        
        zj = scrapy.Selector(response).xpath('//*[@class="bookname"]/h1/text()').extract()[0]
        
        
        item = vbSpiderItem()
        
        item["xs_name_path"] = down_path
        
        item["zhangjie"] = zj
        
        item["zhengwen"] = textstr
        
        yield item
        
        pass
