import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


#不同标注的处理方式
#sl,le,npr,amh,tsc,wa,pl5:true-normalizescaler,fals-minmaxsacler
#sla,dep:true-onehot,false-label
#low_d,n_componets
def hr_preprocessing(sl=False,le=False,npr=False,amh=False,wa=False,
                     pl5=False,sla=False,dep=False,low_d=False,n_com=1):
    #数据清洗
    df=pd.read_csv('../dataset/HR.csv')
    df=df.dropna(subset=['satisfaction_level','last_evaluation'])
    df=df[df['satisfaction_level']<=1][df['salary']!='nme'][df['department']!='sale']
    #标注选择
    label=df['left']
    df=df.drop('left', axis=1)
    #特征选择

    #特征处理
    scaler_lst=[sl,le,npr,amh,wa,pl5]
    column_lst=['satisfaction_level','last_evaluation','number_project',
                'average_monthly_hours','Work_accident','promotion_last_5years']
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            df[column_lst[i]]=MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1,1))#.reshape(1,-1)[0],按index顺序填入的
        else:
            df[column_lst[i]]=StandardScaler().fit_transform(df[column_lst[i]].values.reshape(-1,1))
    scaler_lst=[sla,dep]
    column_lst=['salary','department']
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            if column_lst[i]=='salary':
                df[column_lst[i]]=[map_salary(s) for s in df[column_lst[i]].values]
                #print(map_salary(s for s in df[column_lst[i]].values))
            else:
                df[column_lst[i]]=LabelEncoder().fit_transform(df[column_lst[i]].values.reshape(-1,1))
            df[column_lst[i]]=MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1,1))
        else:
            df=pd.get_dummies(df,columns=[column_lst[i]])
    if low_d:
        return PCA(n_components=n_com).fit_transform(df.values)
    return df,label

d={'low':0,'medium':1,'high':2}
#d=dict([('low',0),('medium',1),('high',2)])


def map_salary(s):
    return d.get(s,0)


def main():
    while 1:
        pd.set_option('display.max_columns', 1024)
        pd.set_option('display.max_rows',2)
        break
    df,label= hr_preprocessing()
    print(df,label)
    #print(df['salary'].value_counts())
if __name__=='__main__':
    main()

