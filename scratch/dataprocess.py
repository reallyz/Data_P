# -- coding:utf-8 --
import  numpy as np
import  pandas as pd
import  os
import xlwt
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel('D:/数据分析进阶路/智联招聘宜宾.xlsx')
salary = data['薪资']

salary = salary.where(salary != '面议').dropna()
salary = salary.where(salary != '1000元/月以下').dropna()
#drop后的数据，index不会自动上移
for index,value in salary.items():
#数据结构items的使用
    n = value.find('-')
    value = value[:n]
    salary[index] = int(value)

#datafram to excel

'''
wbk = xlwt.Workbook()
ws = wbk.add_sheet('salarymodfy4')
tep = range(0,len(salary),1)
print(tep[0],type(tep[0]))
for i in salary.index:
    ws.write(i,0,salary[i]) #不承认指针类型，只承认整型数据
#存在空行
wbk.save('salarymodify4.xls')
'''
#print(salary[salary=='面议'].index,salary[salary=='1000元/月以下'].index)
#sns.countplot(salary)
#sns.set_palette('Reds')
print(type(salary))
#sns.distplot(salary,bins=10)
print(salary)
ss_new=pd.DataFrame({'num':salary.index,'salary':salary.values})
#ss_new.to_excel('./newsalary.xls')
dd=pd.read_excel('./newsalary.xls')

sns.distplot(dd['salary'],bins=10)