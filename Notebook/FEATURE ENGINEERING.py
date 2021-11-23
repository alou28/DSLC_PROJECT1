#!/usr/bin/env python
# coding: utf-8

# # FEATURE ENGINEERING

# ### We are firstly going to see our outliers among our numerical continuous columns 

# In[54]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import plot_confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


data=pd.read_csv("train_prepar.csv")
data1=pd.read_csv("test_prepar.csv")
data.head(10)


# In[33]:


data=data.drop(['Unnamed: 0'],axis=1)


# In[34]:


numerical=[var for var in data.columns if data[var].dtypes!='object']
print("there are {} numericals columns".format(len(numerical)))


# In[35]:


continuous = []#we want to see the number of continous values columns
for var in numerical:
    if len(data[var].unique())>3:
        continuous.append(var)
        
print('There are {} continuous variables'.format(len(continuous)))


# In[36]:


for var in continuous:
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    fig = data.boxplot(column=var)
    fig.set_title('')
    fig.set_ylabel(var)
    
    plt.subplot(1, 2, 2)
    fig = data[var].hist(bins=20)
    fig.set_ylabel('Number of observation')
    fig.set_xlabel(var)

    plt.show()


# In[37]:


data['CODE_GENDER']=data['CODE_GENDER'].replace('XNA','Unknown_Gender')


# In[38]:


data1['CODE_GENDER']=data1['CODE_GENDER'].replace('XNA','Unknown_Gender')


# In[39]:


data['CODE_GENDER'].unique()


# In[40]:


data['TARGET'].unique()


# ### Encode my categorical columns values

# In[41]:


categorical=[var for var in data.columns if data[var].dtypes=='object']


# In[42]:


categorical1=[var for var in data1.columns if data1[var].dtypes=='object']


# In[43]:


for var in categorical:
    print("'{}' has {} labels".format(var,len(data[var].unique())))


# In[44]:


for var in categorical:
    if len(data[var].unique())>2:
        nts=pd.get_dummies(data[var],drop_first=True)
        data=pd.concat([data,nts],axis=1)
        data= data.drop([var],axis=1)
        
    else :
        data[var]=pd.get_dummies(data[var],drop_first=True)
       
        
        
        


# In[45]:


for var in categorical1:
    if len(data1[var].unique())>2:
        nts=pd.get_dummies(data1[var],drop_first=True)
        data1=pd.concat([data1,nts],axis=1)
        data1= data1.drop([var],axis=1)
        
    else :
        data1[var]=pd.get_dummies(data1[var],drop_first=True)


# In[46]:


data['TARGET'].unique()


# In[47]:


data.head()


# ### Scaling of columns with raw data which varies widely

# In[48]:


data['TARGET'].unique()


# In[49]:


#put the value of our continuous variables between 0 and 1 by using min max scaler
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))

for var in continuous:
    data[[var]]=scaler.fit_transform(data[[var]])
    


# In[50]:


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))

for var in continuous:
    data1[[var]]=scaler.fit_transform(data1[[var]])


# ### SEPARATE TRAIN TEST SET

# In[51]:


target_variable =data['TARGET']


# In[52]:


data, data1 = data.align(data1, join = 'inner', axis = 1)


# In[53]:


data['TARGET']=target_variable


# In[55]:


X=data.loc[:, data.columns != 'TARGET']
X=X.drop(['SK_ID_CURR'],axis=1)


# In[56]:


Y=data['TARGET']
Y.unique()


# In[57]:


X.to_csv("train.csv")
Y.to_csv("test.csv")


# In[ ]:




