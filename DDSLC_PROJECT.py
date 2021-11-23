#!/usr/bin/env python
# coding: utf-8

# # Home Credit Default Risk

# # creer un modèle de machine learning qui doit etre capable de prédire si oui ou non un couple peut avoir droit à un pret immobilier.
#NOM:HASSAN BACHACHA,EMERIC BERTIN,ALAIN NGOMEDJ
# In[576]:

""" Importation of libraries"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import seaborn as sns



# # DATA LOADING

# In[577]:


data=pd.read_csv('application_train.csv')
data1=pd.read_csv('application_test.csv')
data.head(10)


# In[578]:


data.shape#taille de mon dataset


# In[579]:


data['TARGET'].unique()


# In[580]:


#we drop columns with more than 65% missing values
def drop_60(df):
    nb_na=df.isna().sum()/df.shape[0]
    v=nb_na[nb_na>0.65]
    l=[i for i in v.index]
    data1=df.drop(l,axis=1)
    return data1


# In[581]:


data=drop_60(data)
data1=drop_60(data1)


# In[582]:


data.shape


# In[583]:


data.describe()


# In[584]:


data.dtypes.value_counts()#The differents types in my dataframes


# In[585]:


def check_na(feature):
    count=0
    list=[]
    for el in feature:
        if data[el].isna().sum()>0:
            list.append(el)
            count+=1
    print(data[list].isna().sum())
    print("{} columns with missing values".format(count))


# In[586]:


categorical=[var for var in data.columns if data[var].dtypes=='object']
categorical1=[var for var in data1.columns if data1[var].dtypes=='object']
categorical


# In[587]:


data.head(5)


# In[588]:


float_=[var for var in data.columns if data[var].dtypes=='float64']
float1_=[var for var in data1.columns if data1[var].dtypes=='float64']


# In[589]:


int_=[var for var in data.columns if data[var].dtypes=='int64']
int1_=[var for var in data1.columns if data1[var].dtypes=='int64']
int_


# In[590]:


data['DAYS_BIRTH'].unique()


# In[591]:


data['Age']=data['DAYS_BIRTH']/(-365)
data1['Age']=data1['DAYS_BIRTH']/(-365)
data['Age'].value_counts()


# In[592]:


data['DAYS_EMPLOYED'].unique()


# In[593]:


data['YEAR_EMPLOYED']=data['DAYS_EMPLOYED']/(-365)
data1['YEAR_EMPLOYED']=data1['DAYS_EMPLOYED']/(-365)
data['YEAR_EMPLOYED'].value_counts()


# In[594]:


#Number of missing value in categorical columns
check_na(categorical)


# In[595]:


check_na(categorical1)


# In[596]:


data['NAME_TYPE_SUITE'].unique()


# In[597]:


data[categorical]=data[categorical].fillna("Unknown"+data[categorical])
data1[categorical1]=data1[categorical1].fillna("Unknown"+data1[categorical1])


# In[598]:


check_na(categorical)


# In[599]:


check_na(float_)


# In[600]:


data[float_]= data[float_].fillna(data[float_].mean())
data1[float1_]= data1[float1_].fillna(data1[float1_].mean())


# In[601]:


check_na(int_)


# In[602]:


data=data.drop(['DAYS_BIRTH','DAYS_EMPLOYED'],axis=1)
data1=data1.drop(['DAYS_BIRTH','DAYS_EMPLOYED'],axis=1)


# In[603]:


data.head()


# In[604]:


data.shape


# In[605]:


cmap=abs(data.corr()['TARGET']).tail(10)
cmap


# In[606]:


data.head()


# In[607]:


sns.set(style="darkgrid")
x=data['TARGET'].value_counts()
sns.barplot(x.index,x.values,alpha=1)
plt.title('THE DISTRIBUTION OF REPAID LOAN OR NOT')
plt.xlabel('TARGET')
plt.ylabel('Number of observations')#most of the make loan have been repaid


# In[608]:


data['NAME_INCOME_TYPE'].unique()


# In[609]:


x=data[data['TARGET']==0]#on essaie d'en apprendre d'avantages sur les personnes qui ont eu à rembourser leurs prets


# In[610]:


target_0=pd.DataFrame(x)
target_0.head()

"""Le crédit renouvelable (revolving loan) est un contrat de prêt permettant de mettre à disposition de l'emprunteur une somme d'argent. Ce crédit est un contrat d'un an renouvelable. Le montant mis à la disposition de l'emprunteur se reconstitue au fur et à mesure de ses remboursements."""

"""un crédit de trésorerie(cash loan) est un crédit à court terme qui peut-être aussi bien accordé aux entreprises qu'aux particuliers. Lorsqu'une entreprise y a recourt, celle-ci pourra disposer temporairement de la trésorerie utile à son fonctionnement. Le remboursement se fera bien entendu à court terme """
# In[611]:




x=target_0['NAME_CONTRACT_TYPE'].value_counts()
sns.barplot(x.index,x.values,alpha=0.9)
plt.title('REPARTITION OF NAME CONTRACT TYPE BASED ON THE REPAID LOAN')
plt.xlabel('NAME CONTRACT TYPE')
plt.ylabel('Number of observations')#the majority of person who repaid their loan have done a cash loan(pret de tresorerie).it means that they already had enough money for their operations 


# In[612]:


from bokeh.transform import cumsum
from bokeh.palettes import viridis
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral6
from bokeh.transform import dodge
from bokeh.palettes import Spectral9
from math import pi


# In[613]:


target_0['CNT_FAM_MEMBERS'].value_counts()


# In[614]:


z=target_0[target_0['NAME_CONTRACT_TYPE']=='Cash loans']
target_1=pd.DataFrame(z)
y=target_1['OCCUPATION_TYPE'].value_counts().head(9)#to know the occupation type of person who repaid loan after take cash loan 


continent = ['Laborers','Sales staff','Core staff','Managers','Drivers','High skill tech staff','Accountants','Medicine staff','Security staff']
counts = target_1.OCCUPATION_TYPE.groupby(target_1.OCCUPATION_TYPE).value_counts().nlargest(10)[1:50000]

source = ColumnDataSource(data=dict(continent=continent, counts=y))

p = figure(x_range=continent, plot_width=1000, title="OCCUPATION TYPE OF PERSON WHO TAKE CASH LOAN FOR REPAIR THEIR LOAN",
           tools='hover',tooltips="@counts")

p.vbar(x='continent', top='counts', width=0.9, source=source, legend="continent",line_color='black', fill_color=factor_cmap('continent', palette=Spectral9, factors=continent))



p.y_range.start = 0
p.legend.location = "top_right"
p.legend.orientation = "horizontal"
show(p)


# In[615]:


x=target_1['NAME_INCOME_TYPE'].head(1000).value_counts()
sns.barplot(x.index,x.values,alpha=0.9)
plt.title('REPARTITION OF NAME INCOME TYPE WHO REPAID LOAN WITH CASH LOAN')
plt.xlabel('NAME INCOME TYPE')
plt.ylabel('Number of observations')


# In[616]:


plt.figure(figsize=(10,5))#Age of persons who repairs their loan
plt.hist(target_1['Age'] , edgecolor = 'orange', bins = 20)
plt.title('Age of Client')
plt.xlabel('Age (years)')
plt.ylabel('Count')


# In[617]:


plt.figure(figsize=(10,5))
plt.xlim(data['AMT_CREDIT'].min(),data['AMT_CREDIT'].max())
plt.xlabel('AMT_CREDIT')
plt.ylabel('Density')
sns.kdeplot(data['AMT_CREDIT'],shade=True)
plt.title("Distribution of AMT_CREDIT")
plt.show()


# In[618]:


x=target_1['CODE_GENDER'].value_counts()
sns.barplot(x.index,x.values,alpha=0.9)
plt.title('REPARTITION OF GENDER WHO REPAID LOAN WITH CASH LOAN')
plt.xlabel('CODE GENDER')
plt.ylabel('Number of observations')#les femmes gagent lus que les hommes


# In[619]:


#les familles moins nombreuses(avec le moins d'enfants) ont donc plus de chance de rembourser leur credit
x=target_1['CNT_FAM_MEMBERS'].value_counts().head(6)
pie, ax = plt.subplots(figsize=[20,10])

plt.pie(x, autopct="%.1f%%", labels=x.index, pctdistance=0.5)
plt.title(" REPAID CASH LOAN BASED ON THE FAMILY NUMBER ", fontsize=15);
pie.savefig("DeliveryPieChart.png")


# In[620]:


#on fera de meme pour name_contract_type=revolved loan puis on remonte a target==1 une prochaine fois


# # FEATURE ENGINEERING

# ### We are firstly going to see our outliers among our numerical continuous columns

# In[621]:


data.head()


# In[622]:


numerical=[var for var in data.columns if data[var].dtypes!='object']
print("there are {} numericals columns".format(len(numerical)))


# In[623]:


continuous = []#we want to see the number of continous values columns
for var in numerical:
    if len(data[var].unique())>3:
        continuous.append(var)
        
print('There are {} continuous variables'.format(len(continuous)))


# In[624]:


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


# In[625]:


data['CODE_GENDER']=data['CODE_GENDER'].replace('XNA','Unknown_Gender')
data1['CODE_GENDER']=data1['CODE_GENDER'].replace('XNA','Unknown_Gender')


# In[626]:


data['CODE_GENDER'].unique()


# In[627]:


data['TARGET'].unique()


# ### Encode my categorical columns values

# In[628]:


for var in categorical:
    print("'{}' has {} labels".format(var,len(data[var].unique())))


# In[629]:


data=data.drop(['ORGANIZATION_TYPE'],axis=1)
data1=data1.drop(['ORGANIZATION_TYPE'],axis=1)
categorical.remove('ORGANIZATION_TYPE')
categorical1.remove('ORGANIZATION_TYPE')


# In[630]:


for var in categorical:
    if len(data[var].unique())>2:
        nts=pd.get_dummies(data[var],drop_first=True)
        data=pd.concat([data,nts],axis=1)
        data= data.drop([var],axis=1)
        
    else :
        data[var]=pd.get_dummies(data[var],drop_first=True)
       
        
        
        


# In[631]:


for var in categorical1:
    if len(data1[var].unique())>2:
        nts=pd.get_dummies(data1[var],drop_first=True)
        data1=pd.concat([data1,nts],axis=1)
        data1= data1.drop([var],axis=1)
        
    else :
        data1[var]=pd.get_dummies(data1[var],drop_first=True)


# In[632]:


data['TARGET'].unique()


# In[633]:


data.head()


# ### Scaling of columns with raw data which  varies widely

# In[634]:


data['TARGET'].unique()


# In[635]:


#put the value of our continuous variables between 0 and 1 by using min max scaler
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))

for var in continuous:
    data[[var]]=scaler.fit_transform(data[[var]])
    


# In[636]:


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))

for var in continuous:
    data1[[var]]=scaler.fit_transform(data1[[var]])


# In[637]:


data.shape


# In[638]:


data['TARGET'].unique()


# In[639]:


data.head(10)


# # SEPARATE TRAIN TEST SET

# In[640]:


target_variable =data['TARGET']


# In[641]:


data, data1 = data.align(data1, join = 'inner', axis = 1)


# In[642]:


data['TARGET']=target_variable


# In[643]:


from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import plot_confusion_matrix


# In[644]:


X=data.loc[:, data.columns != 'TARGET']
X=X.drop(['SK_ID_CURR'],axis=1)


# In[645]:


Y=data['TARGET']
Y.unique()


# In[646]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X_train.shape, X_test.shape


# # MODEL TRAINNING AND PREDICTION

# ### Xgboost Classifier

# In[647]:


import xgboost as xgb
from sklearn.metrics import classification_report
xgb = xgb.XGBClassifier(scale_pos_weight=11,objective="binary:logistic",learning_rate= 0.001, n_estimators= 100)
xgb.fit(X_train, Y_train)


# In[648]:


y_pred=xgb.predict(X_test)
y_pred_auc = xgb.predict_proba(X_test)[:,1]


# In[649]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

roc_score = roc_auc_score(Y_test,y_pred_auc)
print("roc_auc_score: ",roc_score)

acc_score = accuracy_score(Y_test, y_pred)
print("Accuracy_score",acc_score)


# In[650]:


print(classification_report(Y_test, y_pred))


# In[651]:


plot_confusion_matrix(xgb,
                       X_test,
                       Y_test,
                       values_format='d',
                       display_labels=['repaid loan' ,'Did not repaid'])


# ### RandomForest Classifier

# In[652]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 100, random_state = 50, n_jobs = -1,class_weight="balanced")
RF.fit(X_train,Y_train)


# In[654]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

y_pred=RF.predict(X_test)
y_pred_auc =RF.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(Y_test,y_pred_auc)
print("roc_auc_score: ",roc_score)

acc_score = accuracy_score(Y_test,y_pred)
print("Accuracy_score: ",acc_score)


# In[655]:


print(classification_report(Y_test, y_pred))


# ### GradientBossting Classifier

# In[656]:


from sklearn.ensemble import GradientBoostingClassifier


# In[657]:


GB = GradientBoostingClassifier(learning_rate=0.01,n_estimators=100)
GB.fit(X_train, Y_train)


# In[658]:


y_pred=GB.predict(X_test)
y_pred_auc = GB.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(Y_test,y_pred_auc)
print("roc_auc_score: ",roc_score)

acc_score = accuracy_score(Y_test,y_pred)
print("Accuracy_score: ",acc_score)


# # MLFLOW

# #### MLFLOW On xgboost

# In[659]:


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

# In[660]:


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

# In[661]:


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


# In[662]:


train_xgboost(0.0001,100)


# In[663]:


train_random(100)


# In[664]:


train_GB(0.01,50)


# # MODEL EXPLAINATION

# In[665]:


import shap
shap.initjs()


# In[666]:


import eli5
eli5.show_weights(GB)


# In[667]:


rand = X_train.sample(1000, random_state=42)


# In[668]:


explainer = shap.TreeExplainer(GB)


# In[669]:


shap_values = explainer.shap_values(rand)


# In[670]:


feature_names=X_train.columns.tolist()


# In[671]:


shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:])


# In[672]:


shap.dependence_plot("DAYS_REGISTRATION", shap_values, rand)


# In[673]:


shap.summary_plot(shap_values, rand,feature_names= feature_names)


# In[ ]:




