#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

df=pd.read_csv('D:\\depy2016\\adult.csv')
X=df.drop('income',axis=1)
y=df.income
y=[0 if x == '<=50K' else 1 for x in y]
X['age']=X['age'].where(X['age']!='#NAME?')
X['fnlwgt']=X['fnlwgt'].where(X['fnlwgt']!='#NAME?')
X['workclass']=X['workclass'].where(X['workclass']!='?','Never-worked')
X['education']=X['education'].where(X['education']!='?','HS-grad')
X['education_num']=X['education_num'].where(X['education_num']!='#NAME?')
X['occupation']=X['occupation'].where(X['occupation']!='?','Other-service')
X['race']=X['race'].where(X['race']!='#NAME? ','White')
X['sex']=X['sex'].where(X['sex']!='#NAME?','Male')
X['native_country']=['United-States' if x =='United-States' else 'other' for x in X['native_country']]


tep=['age','fnlwgt','education_num']

for i in tep:
    X[i]=X[i].fillna(X[i].median)
#以上为乱码，空值，离散属性降维处理
'''
d=dict()
for col in X.columns:
    d[col]=X[col].dtypes
'''

#TODO
#dummy feature for descrete value
#每个feature+离散属性形成新的feature，衍生属性？
todummy_ls=['workclass','education','marital_status','occupation','relationship',
            'race','sex','native_country']
for x in todummy_ls:
    dummies=pd.DataFrame(pd.get_dummies(X[x],prefix =x,dummy_na=False))
    X=X.drop(x,1)
    X=pd.concat([X,dummies],axis=1)
print(X.head(5))

#TODO
#feature engineering
#属性之间的关联分析？
#简单的two-way interaction:X3=X1*X2,增加新feature，各个feature线性无关时特别有用
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

def add_iteractions(df):
    combos=list(combinations(list(df.columns),2))
    colnames=list(df.columns)+['_'.join(z) for z in combos]
    poly=PolynomialFeatures(interaction_only=True,include_bias=False)
    df=poly.fit_transform(df)
    df=pd.DataFrame(df)
    df.columns = colnames
    noint_indicices=[i for i,x in enumerate(list((df==0).all())) if x ]
    df=df.drop(df.columns[noint_indicices],axis=1)
    return df
X=add_iteractions(X)
print(X.head(2))

#TODO
#PCA降维
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
X_pca = pd.DataFrame(pca.fit_transform(X))

#TODO
#特征选择与回归
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=1)

# Such a large set of features can cause overfitting and also slow computing
# Use feature selection to select the most important features
import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k=20)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]
# Function to build model and find model performance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def find_model_perf(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_hat = [x[1] for x in model.predict_proba(X_test)]
    auc = roc_auc_score(y_test, y_hat)

    return auc

# Find performance of model using preprocessed data
auc_processed = find_model_perf(X_train_selected, y_train, X_test_selected, y_test)
print(auc_processed)