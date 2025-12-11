import pandas as pd
data=pd.read_csv('spam.csv',encoding='utf-8')


data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
data.rename(columns={'v1':'label','v2':'message'},inplace=True)
data.to_csv('clean_spam.csv',index=False)