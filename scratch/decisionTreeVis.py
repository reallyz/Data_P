import pydotplus
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.preprocessing import LabelEncoder
import  pandas as pd
#j决策数的算法属于不纯度或者信息熵
df=pd.read_excel('D:\数据分析进阶路\dataset\smalltree.xlsx')
label=df['PlayTennis']

feature=df.drop(labels=['Day','PlayTennis'],axis=1)
print(type(feature))
col=feature.columns.values
d=dict([('Yes',1),('No',0)])
def map_label(s):
    return d.get(s)
label=[map_label(s) for s in label.values]
for col_name in col:
    feature[col_name]=LabelEncoder().fit_transform(feature[col_name].values.reshape(-1,1))
print(feature)

cls=DecisionTreeClassifier()
cls.fit(feature,label)
data_dot=export_graphviz(cls,out_file=None,feature_names=col,class_names=['No','Yes'],filled=True,rounded=True)
graph=pydotplus.graph_from_dot_data(data_dot)
graph.write_pdf('D:\数据分析进阶路\output\smalltree0-No_1-Yes.pdf')
