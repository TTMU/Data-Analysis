# Airbnb产品数据分析

> 数据处理工具：python
>
> 图像绘制工具：power bi

## 1.背景

**Airbnb**是一个旅行房屋租赁社区，用户可通过网络或手机应用程序发布、搜索度假房屋租赁信息并完成在线预定程序。 据官网显示 以及媒体报道 ，其社区平台在191个国家、65,000个城市为旅行者们提供数以百万计的独特入住选择，不管是公寓、别墅、城堡还是树屋。 **Airbnb**被时代周刊称为“住房中的EBay”。

## 2.分析目的

根据AARRR模型，从以下角度分析问题：

Acquisition：用户群特征有哪些？如何获取更多用户？

Activation：用户激活率是多少？用户激活漏斗模型。

Retention ：留存及复购情况如何？

Revenue/Refer：营销渠道的推广效果如何？当前转化率是多少？

## 3.数据来源

### 来源：

[数据来自来自kaggle](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data)

### 预览：

```python
import numpy as np
import pandas as pd
```

```python
#读取数据并把date_account_created转换为datetime类型
basic_path = r"D:\github\数据集\Airbnb产品"
train_users_df = pd.read_csv(basic_path + r'\train_users.csv',parse_dates=['date_account_created'])
#sessions_df = pd.read_csv(basic_path + r'\sessions.csv')
print(train_users_df.head())
```

```
  id date_account_created  timestamp_first_active date_first_booking  \
0  gxn3p5htnn           2010-06-28          20090319043255                NaN   
1  820tgsjxq7           2011-05-25          20090523174809                NaN   
2  4ft3gnwmtx           2010-09-28          20090609231247         2010-08-02   
3  bjjt8pjhuk           2011-12-05          20091031060129         2012-09-08   
4  87mebub9p4           2010-09-14          20091208061105         2010-02-18   

      gender   age signup_method  signup_flow language affiliate_channel  \

0  -unknown-   NaN      facebook            0       en            direct   
1       MALE  38.0      facebook            0       en               seo   
2     FEMALE  56.0         basic            3       en            direct   
3     FEMALE  42.0      facebook            0       en            direct   
4  -unknown-  41.0         basic            0       en            direct   

  affiliate_provider first_affiliate_tracked signup_app first_device_type  \
0             direct               untracked        Web       Mac Desktop   
1             google               untracked        Web       Mac Desktop   
2             direct               untracked        Web   Windows Desktop   
3             direct               untracked        Web       Mac Desktop   
4             direct               untracked        Web       Mac Desktop   

  first_browser country_destination  
0        Chrome                 NDF  
1        Chrome                 NDF  
2            IE                  US  
3       Firefox               other  
4        Chrome                  US  
```

## 4.数据清洗

### 缺失值查看

```
train_users_df.isnull().sum()
```

```python
id                              0
date_account_created            0
timestamp_first_active          0
date_first_booking         124543
gender                          0
age                         87990
signup_method                   0
signup_flow                     0
language                        0
affiliate_channel               0
affiliate_provider              0
first_affiliate_tracked      6065
signup_app                      0
first_device_type               0
first_browser                   0
country_destination             0
dtype: int64
```

date_first_booking，age，first_affiliate_tracked三列含有缺失值。

### 异常值查看

```python
train_users_df.describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp_first_active</th>
      <th>age</th>
      <th>signup_flow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.134510e+05</td>
      <td>125461.000000</td>
      <td>213451.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.013085e+13</td>
      <td>49.668335</td>
      <td>3.267387</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.253717e+09</td>
      <td>155.666612</td>
      <td>7.637707</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.009032e+13</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.012123e+13</td>
      <td>28.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.013091e+13</td>
      <td>34.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.014031e+13</td>
      <td>43.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.014063e+13</td>
      <td>2014.000000</td>
      <td>25.000000</td>
    </tr>
  </tbody>
</table>

age 的最大值是2014，不符合常识，猜测错把出生日期当成年龄填入，需要处理掉。将年龄数据小于0大于100的数据修改为 -1,转换数据为int型，方便后续分类统计。

```python
train_users_df.loc[(pd.isnull(train_users_df['age']))|(train_users_df['age']>100),'age']= -1
train_users_df['age'] = train_users_df['age'].astype('int')
```

### 重复值查看

根据数据要求，表中id应唯一，对key：id 进行去重。

```python
train_users_df = train_users_df.drop_duplicates(subset=['id'],keep='first')
```

## 5.数据分析

### 5.1用户画像分析

1. 性别分布

   ```python
   gender_df = train_users_df[(train_users_df['gender']=='FEMALE')|(train_users_df['gender']== 'MALE')].groupby(by = 'gender').size()
   ```

   | 性别   | 人数  |
   | ------ | ----- |
   | FEMALE | 63041 |
   | MALE   | 54440 |

   ![性别分布](性别分布.png)

   

2. 年龄分布

   ```python
   def change_age(age):
       if age <= 20:
           return "20岁以下"
       elif age >20 and age <=30:
           return "20到30岁"
       elif age >30 and age <=40:
           return "30到40岁"
       elif age >40 and age <=50:
           return "40到50岁"
       else:
           return "50岁以上"
           
   age_df = train_users_df.copy()
   age_df['age'] = age_df["age"].apply(lambda x:change_age(x))
   age_df = age_df.groupby("age").size()
   ```

   | 年龄     | 人数  |
   | -------- | ----- |
   | 20岁以下 | 92804 |
   | 20到30岁 | 41633 |
   | 30到40岁 | 44212 |
   | 40到50岁 | 18831 |
   | 50岁以上 | 15971 |

   ![性别分布](年龄分布.png)

3. 使用语言分布

   ```python
   language_df = train_users_df.groupby('language').size().sort_values(ascending=False)
   ```

   | 语言 | 人数   |
   | ---- | ------ |
   | en   | 206314 |
   | zh   | 1632   |
   | fr   | 1172   |
   | es   | 915    |
   | ko   | 747    |
   | de   | 732    |
   | it   | 514    |
   | ru   | 389    |
   | pt   | 240    |
   | ja   | 225    |
   | sv   | 122    |
   | nl   | 97     |
   | tr   | 64     |
   | da   | 58     |
   | pl   | 54     |
   | cs   | 32     |
   | no   | 30     |
   | th   | 24     |
   | el   | 24     |
   | id   | 22     |
   | hu   | 18     |
   | fi   | 14     |
   | is   | 5      |
   | ca   | 5      |
   | hr   | 2      |

   ![性别分布](语言分布.png)

4. 国家目的地分布

   ```python
   country_df = train_users_df[train_users_df['country_destination'] != 'NDF'].groupby('country_destination').size().sort_values(ascending=False)
   ```

   | 国家  | 人数  |
   | ----- | ----- |
   | US    | 62376 |
   | other | 10094 |
   | FR    | 5023  |
   | IT    | 2835  |
   | GB    | 2324  |
   | ES    | 2249  |
   | CA    | 1428  |
   | DE    | 1061  |
   | NL    | 762   |
   | AU    | 539   |
   | PT    | 217   |

   ![目的地国家](目的地国家.png)

5. 注册app来源分布

   ```python
   sign_app_df = train_users_df[train_users_df['signup_app'] != 'NDF'].groupby('signup_app').size().sort_values(ascending=False)
   
   web_device_type_df = train_users_df[train_users_df['signup_app'] == 'Web'].groupby('first_device_type').size().sort_values(ascending=False)
   ```

   | 方式    | 人数   |
   | ------- | ------ |
   | Web     | 182717 |
   | iOS     | 19019  |
   | Moweb   | 6261   |
   | Android | 5454   |

   | web细分            | 人数  |
   | ------------------ | ----- |
   | Mac_Desktop        | 86839 |
   | Windows_Desktop    | 70887 |
   | iPad               | 13033 |
   | Other/Unknown      | 4773  |
   | iPhone             | 4050  |
   | Desktop_(Other)    | 1145  |
   | Android_Tablet     | 1096  |
   | Android_Phone      | 842   |
   | SmartPhone_(Other) | 52    |

   ![](注册app.png)

   

### 5.2流量分析

1. 用户量变化

   ```
   train_users_df['year_month'] = train_users_df['date_account_created'].apply(lambda x : x.strftime('%Y-%m'))
   account_crea_df = train_users_df.groupby("year_month").size()
   ```

   > 篇幅所限，节选10行数据

   | 时间    | 人数 |
   | ------- | ---- |
   | 2010-01 | 61   |
   | 2010-02 | 102  |
   | 2010-03 | 163  |
   | 2010-04 | 157  |
   | 2010-05 | 227  |
   | 2010-06 | 222  |
   | 2010-07 | 307  |
   | 2010-08 | 312  |
   | 2010-09 | 371  |

   ![](用户量.png)



2.渠道转化

```python
af_prv_df = train_users_df[(train_users_df['date_first_booking'].notnull())].groupby(['affiliate_channel','affiliate_provider']).size().to_frame()
af_prv_df_1 = train_users_df.groupby(['affiliate_channel','affiliate_provider']).size().to_frame()
af_prv_df = af_prv_df.merge(af_prv_df_1, how='inner', left_index=True , right_index=True).reset_index()
af_prv_df.columns = ['channel','provider','渠道预定数量','渠道全部数量']
af_prv_df = af_prv_df[af_prv_df['渠道预定数量'] > 200]
af_prv_df['类型'] = af_prv_df['channel'] + '-' + af_prv_df['provider']
af_prv_df['占比'] = af_prv_df['渠道预定数量'] /af_prv_df['渠道全部数量']
```

![](渠道转化.png)

3.营销广告转化

```python
first_aff_trac_df = train_users_df[(train_users_df['date_first_booking'].notnull())].groupby(['first_affiliate_tracked']).size().to_frame()
first_aff_trac_df_1 = train_users_df.groupby(['first_affiliate_tracked']).size().to_frame()
first_aff_trac_df = first_aff_trac_df.merge(first_aff_trac_df_1, how='inner', left_index=True , right_index=True).reset_index()
first_aff_trac_df.columns = ['营销广告来源','该来源预定人数','该来源全部人数']
#first_aff_trac_df = first_aff_trac_df[first_aff_trac_df['渠道预定数量'] > 200]
#first_aff_trac_df['类型'] = first_aff_trac_df['channel'] + '-' + first_aff_trac_df['provider']
first_aff_trac_df['占比'] = first_aff_trac_df['该来源预定人数'] /first_aff_trac_df['该来源全部人数']
first_aff_trac_df = first_aff_trac_df[first_aff_trac_df['营销广告来源'] != 'untracked']
```

![](广告占比.png)

4.转化量漏斗分析

```python
#总用户量
total_user_df = sessions_df.drop_duplicates('user_id')
#注册用户数
user_creat_df = total_user_df[['user_id','action_detail']].merge(train_users_df[['id','date_account_created']],how = 'left',left_on = 'user_id',right_on = 'id')
#定义用户在平台浏览次数超过10次即为活跃用户。活跃用户数为
n_df = sessions_df[['user_id','action_detail']].merge(train_users_df[['id','date_account_created']],how = 'left',left_on = 'user_id',right_on = 'id')
n_df = n_df[n_df['date_account_created'].notnull()]
n_df = n_df.groupby('user_id').size().to_frame().reset_index()
n_df.columns = ['user_id','num']
n_df = n_df[n_df['num']>10]
#注册用户中提交过订单信息的用户数
place_order_df = user_creat_df[user_creat_df['date_account_created'].notnull()]
place_order_df = sessions_df[sessions_df['action_detail'] == 'reservations'].groupby('user_id').size()
#注册用户中成功支付的用户数
place_order_df = user_creat_df[user_creat_df['date_account_created'].notnull()]
place_order_df = sessions_df[sessions_df['action_detail'] == 'payment_instruments'].groupby('user_id').size()
#重复购买用户数
re_pay_user_df = place_order_df.to_frame().reset_index()
re_pay_user_df.columns = ["user_id","num"]
re_pay_user_df = re_pay_user_df[re_pay_user_df['num'] > 1]
	
```

| 类别             | 人数   |
| ---------------- | ------ |
| 总用户量         | 135484 |
| 注册用户数       | 73815  |
| 活跃用户数       | 58881  |
| 提交订单用户数   | 10366  |
| 成功支付的用户数 | 9018   |
| 重复购买用户数   | 4153   |

![](转化率.png)

## 6.总结