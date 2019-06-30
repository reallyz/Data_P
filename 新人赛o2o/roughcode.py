

import pandas as pd
import numpy as np
pd.set_option('Max_columns',1024)
'''
offline=pd.read_csv('./ccf_offline_stage1_train/ccf_offline_stage1_train.csv',usecols=['User_id','Merchant_id'])
online=pd.read_csv('./ccf_online_stage1_train/ccf_online_stage0_train.csv',usecols=['User_id','Merchant_id'])
def isidentical(x):
    diff=offline[x]/online[x]
    count=0
    for t in diff:
        if t==1:
            count+=1
    if count:
        return print('has identical rows:',count)
    else:
        return print('not identical')

isidentical('User_id')
isidentical('Merchant_id')
线上数据和线下不统一
'''
dft=pd.read_csv('./ccf_offline_stage1_train/ccf_offline_stage1_train.csv',nrows=2000)


def label(df):
    if pd.to_datetime(df['Date'],format='%Y%m%d')-pd.to_datetime(df['Date_received'],format='%Y%m%d') <= pd.Timedelta(15,'D'):
        return 1
    else:
        return 0


def getrate(row):
    if pd.isnull(row):
        return 0
    elif ':' in row:
        rows=row.split(':')
        return 1-float(rows[1])/float(rows[0])
    else:
        return float(row)


def getman(row):
    if ':' in str(row):
        return row.split(':')[0]
    else:
        return 0


def getjian(row):
    if ':' in str(row):
        return row.split(':')[1]
    else:
        return 0

def getweekday(row):
    if pd.isnull(row):
        return row
    else:
        return pd.to_datetime(row,format='%Y%m%d').weekday()


dft['label']=dft.apply(label,axis=1)
dft['getrate']=dft['Discount_rate'].apply(getrate)
dft['getman']=dft['Discount_rate'].apply(getman)
dft['getjian']=dft['Discount_rate'].apply(getjian)
dft['weekdayR']=dft['Date_received'].apply(getweekday)
dft['weekdayS']=dft['Date'].apply(getweekday)
print(dft.head(10))
