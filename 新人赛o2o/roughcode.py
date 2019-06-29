

import pandas as pd
import numpy as np

dft=pd.read_csv('./ccf_offline_stage1_train/ccf_offline_stage1_train.csv',nrows=10000)

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

dft['getrate']=dft['Discount_rate'].apply(getrate)
dft['getman']=dft['Discount_rate'].apply(getman)
dft['getjian']=dft['Discount_rate'].apply(getjian)
dft['label']=dft.apply(label,axis=1)
pd.set_option('Max_columns',1024)
print(dft.head(10))