

import pandas as pd

df=pd.read_csv('./ccf_offline_stage1_train/ccf_offline_stage1_train.csv')

def label(df):
    if pd.to_datetime(df['Date'],format='%Y%m%d')-pd.to_datetime(df['Date_received'],format='%Y%m%d') <= pd.Timedelta(15,'D'):
        return 1
    else:
        return 0
df['label']=df.apply(label,axis=1)
print(df['label'].head(20))
print(df['label'].value_counts())