import numpy as np
from sklearn.decomposition import  PCA
import pandas as pd

#TODO
#降维
#PCA
data = np.array([np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1]),
                np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9])]).T
data1 = np.array([[2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1],[2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9]]).T
#skearn pca method 奇异值分解
lower_dim =PCA(n_components=1)
lower_dim.fit(data)
print('information ratio:\n',lower_dim.explained_variance_ratio_)
print('transformed data:\n',lower_dim.transform(data))
#主成分分析
def myPCA(data,n_componets=10000):
    mean_vals=np.mean(data,axis=0)
    mid=data-mean_vals
    cov_mat=np.cov(mid,rowvar=False)
    from scipy import linalg
    eig_val,eig_vec=linalg.eig(np.mat(cov_mat))
    eig_val_index=np.argsort(eig_val)
    eig_val_index=eig_val_index[:-(n_componets+1):-1]
    eig_vec=eig_vec[:,eig_val_index]
    low_dim_mat=np.dot(mid,eig_vec)
    return  low_dim_mat,eig_val

print(myPCA(data,n_componets=1)[0],'\n',myPCA(data,n_componets=1)[1])

#TODO
#离散值相关性分析（熵的计算）
s1=pd.Series(['x1','x1','x2','x2','x2','x2'])
s2=pd.Series(['y1','y1','y1','y2','y2','y2'])

def getEntropy(s):
    if not isinstance(s,pd.core.series.Series):
        s=pd.Series(s)
    s_dbt=s.groupby(s).count().values
    p=s_dbt/float(len(s))
    return -sum(p*np.log2(p))

print(getEntropy(s1))
def getCond_Entropy(s1,s2):
    d=dict()
    for i in list(range(len(s1))):
        d[s1[i]]=d.get(s1[i],[])+[s2[i]]
    sum=0
    for key in d.keys():
        tep=getEntropy(d[key])*len(d[key])/float(len(s1))
        sum=sum+tep
    return sum
print(getCond_Entropy(s1,s2))

def getEntropyGain(s1,s2):
    return getEntropy(s2)-getCond_Entropy(s1,s2)
print(getEntropyGain(s1,s2))


def getEntropyGainRatio(s1,s2):
    return getEntropyGain(s1,s2)/getEntropy(s2)
print(getEntropyGainRatio(s1,s2))

def getCorrEntropy(s1,s2):
    return getEntropyGain(s1,s2)/np.sqrt(getEntropy(s1)*getEntropy(s2))
print(getCorrEntropy(s2,s1))
#TODO
#gini系数的计算
def getPross(s):
    if not isinstance(s,pd.core.series.Series):
        s=pd.Series(s)
    p=s.groupby(s).count().values/float(len(s))
    return sum(p**2)
def getGini(s1,s2):
    d=dict()
    for i in list(range(len(s1))):
        d[s1[i]]=d.get(s1[i],[])+[s2[i]]
    return 1-sum([getPross(d[key])*len(d[key])/float(len(s1)) for key in d])
print(getGini(s2,s1))







