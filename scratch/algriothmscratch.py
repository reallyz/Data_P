import numpy as np
from sklearn.decomposition import  PCA
import pandas as pd
import  random as rd
from sklearn import preprocessing
import matplotlib.pyplot as plt

#TODO
#概率与数理统计基本概念
#概率中的相关指的是线性相关
X = np.array([[0,0,0],[1,0,1],[1,0,0],[1,1,0]]).T
X_mean=np.mean(X,1)
E=np.zeros([len(X),len(X)])
for i in range(len(X)):
    for j in range(len(X)):
        E[i,j]=E[j,i]=(X[i]-X_mean[i]).dot(X[j]-X_mean[j])/len(X[i])

print(E,'\n')
print('np.cov(x,bias=1):\n{}\n\n'.format(np.cov(X,bias=1)))
print('np.cov(x,bias=0):\n{}\n\n'.format(np.cov(X)))


#TODO
#协方差


#TODO
#降维
#introduce
#create data
plt.rcParams['axes.unicode_minus']=False
genes=['gene'+str(i) for i in range(1,101)]
wt=['wt'+str(i) for i in range(1,6)]
ko=['ko'+str(i) for i in range(1,6)]
datamk=pd.DataFrame(columns=[*wt,*ko],index=genes)
for gene in datamk.index:
    datamk.loc[gene,'wt1':'wt5']=np.random.poisson(lam=rd.randrange(10,1000),size=5)
    datamk.loc[gene,'ko1':'ko5']=np.random.poisson(lam=rd.randrange(10,1000),size=5)
#center and scale data,after centering the average value for each gene will be 0
#after sacling,the standard deviation for the values for each gene will be 1
scaled_data=preprocessing.scale(datamk.T) #scale function expect samples(compare to varibles) to be rows instead of columns
# or StandardScaler().fit_transform(datamk,T)
pca=PCA()#create PCA object
pca.fit(scaled_data)#do the PCA math
pca_data=pca.fit_transform(scaled_data)#generate coordinates for PCA graph
#scree plot
per_var=np.round(pca.explained_variance_ratio_*100,decimals=1) #calculate the percentage of variation that each principal
#component accounts for
labels=['PC'+str(i) for i in range(1,len(per_var)+1)]
plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
plt.xlabel('Principal Component')
plt.ylabel('Percentage of Explained Variance')
plt.title('Scree Plot')

#starat to draw PCA plot
plt.figure('PCA Graph')
pca_df = pd.DataFrame(pca_data,index=[*wt,*ko],columns=labels)
plt.scatter(pca_df.PC1,pca_df.PC2)
plt.title('PCA Graph')
plt.xlabel('PC1-{}%'.format(per_var[0]))
plt.ylabel('PC2-{}%'.format(per_var[1]))
for sample in pca_df.index:
    plt.annotate(sample,(pca_df.PC1.loc[sample],pca_df.PC2.loc[sample]))
#show the loading scores for PC1 to determine which genes
#had the largest influence on separating the two clusters along the x-axis
loading_scores=pd.Series(pca.components_[0],index=genes)
sorted_loading_scores=loading_scores.abs().sort_values(ascending=False)
top_10_genes=sorted_loading_scores[0:10].index.values
print(loading_scores[top_10_genes])

plt.show()



#PCA
data = np.array([np.array([2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9]),np.array([2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
                ]).T
data1 = np.array([[2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1],[2.4,0.7,2.9,2.2,3,2.7,1.6,1.1,1.6,0.9]]).T
#skearn pca method 奇异值分解
data_scaled=preprocessing.scale(data)
lower_dim =PCA(n_components=2)
lower_dim.fit(data)
print('information ratio:\n',lower_dim.explained_variance_ratio_)
print('transformed data:\n',lower_dim.transform(data))
print('fit_transform:\n',lower_dim.fit_transform(data),'\n\n')
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







