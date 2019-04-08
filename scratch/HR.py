import pandas as pd
import seaborn as sns
import numpy as np
import  scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


plt.rcParams['axes.unicode_minus']=False
df=pd.read_csv('../dataset/hr.csv')
df=df.drop(axis=1,index=[14999,15000,15001])
#TODO
#交叉分析，离职率的按部门分布是否有显著差异
#分布检验，透视表
#独立t分布
dfg=df.groupby(by='department').indices#取的是integer而不是index
itl=df['left'].loc[dfg['IT']].values
Rndl=df['left'].loc[dfg['RandD']].values
#print(ss.ttest_ind(itl,Rndl)[1])
list_dfg=list(dfg.keys())
mats=np.zeros([len(list_dfg),len(list_dfg)])
for i in range(len(list_dfg)):
    for j in range (len(list_dfg)):
        p_val=ss.ttest_ind(df['left'].iloc[dfg[list_dfg[i]]].values,#避免index值与不一致
                           df['left'].iloc[dfg[list_dfg[j]]].values)[1]
        if p_val<0.05:
            mats[i][j]=-1
        else:
            mats[i][j]=p_val
plt.figure('ttest')
sns.heatmap(mats,xticklabels=list_dfg,yticklabels=list_dfg)
plt.savefig('./ttest.png')
plt.figure('pivot')
piv_tb=pd.pivot_table(df,values='left',index=['promotion_last_5years','salary']
                      ,columns='Work_accident')
sns.heatmap(piv_tb,cmap=sns.color_palette('Reds'))
plt.savefig('./pivotimage.png')

#TODO
#分组分析，连续值的分组分析：导数？gini系数，聚类分析
#利用seaborn 画出图形
#离散值，向下钻取
plt.figure('离散值分组钻取')
sns.barplot(x='salary',y='left',hue='department',data=df)
#连续值分离
'''
plt.figure('连续值分离')
stf_l=df['satisfaction_level']
sns.barplot(list(range(len(df))),stf_l.sort_values())
'''
#TODO
#相关性分析
plt.figure('相关性分析')
sns.heatmap(df.corr(),vmin=-1,vmax=1)

#TODO
#因子分析（成分分析）
#解释模糊，需要查文档
plt.figure('因子分析')
myPca=PCA(n_components=7)
low_mat=myPca.fit_transform(df.drop(labels=['salary','department','left'],axis=1))
print(myPca.explained_variance_ratio_)
low_mat_d=pd.DataFrame(low_mat)
sns.heatmap(low_mat_d.corr(),vmin=-1,vmax=1)

plt.show()



