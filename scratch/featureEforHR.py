import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import pydotplus



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
    features=df
    return features,label

d={'low':0,'medium':1,'high':2}
#d=dict([('low',0),('medium',1),('high',2)])


def map_salary(s):
    return d.get(s,0)
def hr_modeling(features,label):
    from sklearn.model_selection import train_test_split
    X_tt,X_validation,Y_tt,Y_validation=train_test_split(features.values,label.values,test_size=0.2)
    X_train,X_tt,Y_train,Y_tt=train_test_split(X_tt,Y_tt,test_size=0.25)
    print(len(X_train),len(X_tt),len(X_validation))
#TODO
#KNN分类器
    from sklearn.metrics import accuracy_score, recall_score, f1_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB,BernoulliNB
    from sklearn.tree import DecisionTreeClassifier,export_graphviz
    models=[]
    models.append(('KNN',KNeighborsClassifier()))
    models.append(('GNB',GaussianNB()))
    models.append(('BNB',BernoulliNB()))
    models.append(('DesT',DecisionTreeClassifier()))
    dataset=[(X_train,Y_train),(X_validation,Y_validation),(X_tt,Y_tt)]
    lis_n=['train','validation','test']
    for cls_name,cls in models:
        print(cls_name,'\n','-'*20)
        cls.fit(X_train,Y_train)
        if cls_name=='DesT':
            data_dot=export_graphviz(cls,out_file=None,feature_names=features.columns.values,
                                     class_names=['NL','L'],filled=True,rounded=True,special_characters=True)
            graph=pydotplus.graph_from_dot_data(data_dot)
            graph.write_pdf('../output/decision_tree_noname.pdf')

        for i in range(len(dataset)):
            print(lis_n[i],'\n','+'*20)
            Y_pred=cls.predict(dataset[i][0])
            Y_t=dataset[i][1]
            print('auc:\n',accuracy_score(Y_t,Y_pred))
            print('roc:\n',recall_score(Y_t,Y_pred))
            print('f1:\n',f1_score(Y_t,Y_pred))



def main():
    while 1:
        pd.set_option('display.max_columns', 1024)
        pd.set_option('display.max_rows',2)
        break
    features,label= hr_preprocessing(dep=True)
    hr_modeling(features,label)
    #print(features,label)
    #print(df['salary'].value_counts())
if __name__=='__main__':
    main()

