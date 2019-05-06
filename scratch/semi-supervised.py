import numpy as np
from  sklearn import datasets
from sklearn.metrics import accuracy_score,f1_score,recall_score
from sklearn.semi_supervised import LabelPropagation

iris=datasets.load_iris()
labels=iris.target
rup=np.random.rand(len(labels))
rup=rup<0.7
Y=labels[rup]
labels[rup]=-1
print('unlabeled_points:',list(labels).count(-1))
lp=LabelPropagation()
lp.fit(iris.data,labels)
y_p=lp.predict(iris.data)
y_pt=y_p[rup]
print('ACC:',accuracy_score(Y,y_pt))
print('ROC:',recall_score(Y,y_pt,average='micro'))
print('F1:',f1_score(Y,y_pt,average='micro'))
print('*8**'*20)
print('ACC:',accuracy_score(iris.target,y_p))
print('ROC:',recall_score(iris.target,y_p,average='micro'))
print('F1:',f1_score(iris.target,y_p,average='micro'))
