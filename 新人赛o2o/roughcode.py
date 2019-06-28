

import pandas as pd

df=pd.read_csv('./ccf_offline_stage1_train/ccf_offline_stage1_train.csv',date_parser=['Date_received','Date'])
coup=df['Coupon_id']