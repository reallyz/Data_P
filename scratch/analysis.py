# -- coding:utf-8 --
import  numpy  as np
import  pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import time
import  seaborn as sns


#font_set = FontProperties(fname=r"c:\windows\fonts\simhei.ttf", size=12)
#中文字体不显示
data = pd.read_excel('D:\\数据分析进阶路\\智联招聘成都1.xlsx')
degree = data['最低学历']
company = data['公司性质']
salary = data['薪资']

lbs = degree.value_counts().index
f = plt.figure()
plt.pie(degree.value_counts(normalize=True),labels=lbs,autopct='%1.1f%%')
plt.savefig('D:\\数据分析进阶路\degreecd1.png')
lbsc = company.value_counts().index
f1 = plt.figure()
plt.pie(company.value_counts(normalize=True),labels=lbsc,autopct='%1.1f%%')
plt.savefig('D:\\数据分析进阶路\companycd1.png')
f2 = plt.figure()

plt.show()

