#!/usr/bin/env python
# coding: utf-8

# # MODEL TRAINNING 

# In[7]:


import pandas as pd
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


# ### xgboost 

# In[8]:


X=pd.read_csv("train.csv")
Y=pd.read_csv("test.csv")


# In[9]:


X=X.drop(['Unnamed: 0'],axis=1)
Y=Y.drop(['Unnamed: 0'],axis=1)


# In[11]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X_train.shape, X_test.shape


# ### MODEL TRAINNING AND PREDICTION

# ### Xgboost Classifier

# In[12]:


import xgboost as xgb
from sklearn.metrics import classification_report
xgb = xgb.XGBClassifier(scale_pos_weight=11,objective="binary:logistic",learning_rate= 0.001, n_estimators= 100)
xgb.fit(X_train, Y_train)


# In[13]:


y_pred=xgb.predict(X_test)
y_pred_auc = xgb.predict_proba(X_test)[:,1]


# In[14]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

roc_score = roc_auc_score(Y_test,y_pred_auc)
print("roc_auc_score: ",roc_score)

acc_score = accuracy_score(Y_test, y_pred)
print("Accuracy_score",acc_score)


# In[15]:


print(classification_report(Y_test, y_pred))


# In[16]:


plot_confusion_matrix(xgb,
                       X_test,
                       Y_test,
                       values_format='d',
                       display_labels=['repaid loan' ,'Did not repaid'])


# ### RandomForest Classifier

# In[17]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 100, random_state = 50, n_jobs = -1,class_weight="balanced")
RF.fit(X_train,Y_train)


# In[18]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

y_pred=RF.predict(X_test)
y_pred_auc =RF.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(Y_test,y_pred_auc)
print("roc_auc_score: ",roc_score)

acc_score = accuracy_score(Y_test,y_pred)
print("Accuracy_score: ",acc_score)


# In[19]:


print(classification_report(Y_test, y_pred))


# ### GradientBossting Classifier 

# In[20]:


from sklearn.ensemble import GradientBoostingClassifier


# In[21]:


GB = GradientBoostingClassifier(learning_rate=0.01,n_estimators=100)
GB.fit(X_train, Y_train)


# In[22]:


y_pred=GB.predict(X_test)
y_pred_auc = GB.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(Y_test,y_pred_auc)
print("roc_auc_score: ",roc_score)

acc_score = accuracy_score(Y_test,y_pred)
print("Accuracy_score: ",acc_score)


# # MLFLOW

# #### MLFLOW On xgboost

# In[23]:


import xgboost as xgb
from xgboost import XGBClassifier
from mlflow.utils.environment import _mlflow_conda_env
import os
import warnings
import sys
import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score

def train_xgboost(lr, n_estim):
    
    with mlflow.start_run(run_name='xgboost'):
               
        xgb =XGBClassifier(learning_rate= lr, n_estimators= n_estim, seed= 42, subsample= 1, colsample_bytree= 1,objective= 'binary:logistic',max_depth= 3,scale_pos_weight=11)
        xgb.fit(X_train, Y_train)
        mlflow.xgboost.autolog()
        
        y_pred_auc = xgb.predict_proba(X_test)[:,1]
        y_pred = xgb.predict(X_test)

       
        roc = roc_auc_score(Y_test, y_pred_auc)*100
        acc = accuracy_score(Y_test,y_pred)
       
        mlflow.log_metric("auc_roc",roc)
        mlflow.log_metric("accuracy_score",acc)
       
        
        mlflow.log_param('learning_rate', lr)
        mlflow.log_param('n_estimators', n_estim)
        mlflow.log_param('seed', 0)
        mlflow.log_param('subsample', 1)
        mlflow.log_param('colsamples_bytree', 1)
        mlflow.log_param('objective','binary:logistic')
        mlflow.log_param('max_depth', 3)
        mlflow.log_param('scale_pos_weight', 11)


        #log model
        mlflow.xgboost.log_model(xgb, "model")
        print("roc_auc",roc)
        print("accuracy_score",acc)
        


# #### MLFLOW ON Random Forest 

# In[24]:


import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def train_random(numb_est):
     with mlflow.start_run(run_name='random forest'):
        
      
        RF = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1,class_weight="balanced")
        RF.fit(X_train,Y_train)
      
        mlflow.xgboost.autolog()
      
        
        y_pred_auc = RF.predict_proba(X_test)[:,1]
        y_pred = RF.predict(X_test)
        
      
       
       
        roc_auc = roc_auc_score(Y_test, y_pred_auc)*100
        acc = accuracy_score(Y_test,y_pred)
        
        
        mlflow.log_metric("auc_roc",roc_auc)
        mlflow.log_metric("accuracy_score",acc)
        
       
        
       
        mlflow.log_param('n_estimators', numb_est)
        mlflow.log_param('random_state', 50)
        mlflow.log_param('verbose', 1)
        mlflow.log_param('n_jobs', -1)
        mlflow.log_param('class_weight','balanced')
        


        # log model
        mlflow.sklearn.log_model(RF, "model")
        print("roc_auc",roc_auc)
        print("accuracy_score",acc)


# #### MLFLOW ON GradientBoosting 

# In[25]:


import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def train_GB(lr, numb_est):
    with mlflow.start_run(run_name ='GrandientBoosting'):
        
        
        gbc = GradientBoostingClassifier(learning_rate = lr, n_estimators = numb_est, random_state=50)
        gbc.fit(X_train, Y_train)
      
        mlflow.xgboost.autolog()
      
        
        y_pred_auc = gbc.predict_proba(X_test)[:,1]
        y_pred = gbc.predict(X_test)

      
       
       
        roc = roc_auc_score(Y_test, y_pred_auc)*100
        acc = accuracy_score(Y_test,y_pred)
        
        
        mlflow.log_metric("auc_roc",roc)
        mlflow.log_metric("accuracy_score",acc)
        
       
        
        mlflow.log_param('learning_rate', lr)
        mlflow.log_param('n_estimators', numb_est)
        mlflow.log_param('random_state', 50)
        
        
        


        # log model
        mlflow.sklearn.log_model(gbc, "model")
        print("roc_auc",roc)
        print("accuracy_score",acc)


# In[26]:


train_xgboost(0.0001,100)


# In[27]:


train_random(100)


# In[28]:


train_GB(0.01,50)


# # MODEL EXPLAINATION

# In[29]:


import shap
shap.initjs()


# In[30]:


import eli5
eli5.show_weights(G)


# In[31]:


rand = X_train.sample(1000, random_state=42)


# In[32]:


explainer = shap.TreeExplainer(GB)


# In[33]:


shap_values = explainer.shap_values(rand)


# In[34]:


feature_names=X_train.columns.tolist()


# In[35]:


shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:])


# In[36]:


shap.dependence_plot("DAYS_REGISTRATION", shap_values, rand)


# In[37]:


shap.summary_plot(shap_values, rand,feature_names= feature_names)


# In[ ]:




