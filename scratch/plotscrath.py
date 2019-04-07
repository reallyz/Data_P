import seaborn as sns

#practice plot
import pandas as pd
import seaborn as sns
import numpy as np
import  scipy.stats as ss


#lmplot
df=sns.load_dataset('anscombe')
sns.set(style='ticks')
sns.lmplot(x='x',y='y',col='dataset',hue='dataset',data=df,
           col_wrap=2)