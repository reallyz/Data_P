import numpy as np
import  matplotlib.pyplot as plt
from sklearn.datasets import make_circles,make_blobs,make_moons
from sklearn.cluster import KMeans

n_s=1000
circles=make_circles(n_samples=n_s,factor=0.5,noise=0.05)
moons=make_moons(n_samples=n_s,noise=0.05)
blobs=make_blobs(n_samples=n_s,random_state=8)
random_data=np.random.rand(n_s,2),None
#t=None,None,type(t)==tuple
colors='bgrcmyk'
data=[circles,moons,blobs,random_data]
models=[('None',None),('Kmeans',KMeans(n_clusters=2))]
f=plt.figure()
plt.rcParams['axes.unicode_minus']=False
for index,clt in enumerate(models):
    clt_name,clt_entity=clt
    for indexs,dataset in enumerate(data):
        X,Y=dataset
        if not clt_entity:
            clt_res=[0 for i in range(len(X))]
        else:
            clt_entity.fit(X)
            clt_res=clt_entity.labels_.astype(np.int)
        plt.title(clt_name)
        f.add_subplot(len(models),len(data),index*len(data)+indexs+1)
        plt.title(clt_name)
        [plt.scatter(X[p,0],X[p,1],color=colors[clt_res[p]]) for p in range(len(X))]
plt.show()
