# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class vbSpiderItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    xs_name_path = scrapy.Field()
    zhangjie = scrapy.Field()
    zhengwen = scrapy.Field() 
    pass
