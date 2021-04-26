#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 12:01:44 2021

@author: mutt
"""

from enum import Enum , unique




@unique
class url_spy(Enum):
    vbiquge = "vbiquge_spy"
    
    
    
txt_url = "www.vbiquge.com/20_20331/"   #爬取书主目录



from xs_spider.spiders.vbiquge_spy import VbiqugeSpySpider

from twisted.internet import reactor
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings



runner = CrawlerRunner(get_project_settings())

d = runner.crawl('vbiquge_spy', search_url = 'https://www.vbiquge.com/20_20331/')

d.addBoth(lambda _: reactor.stop())

reactor.run() 


