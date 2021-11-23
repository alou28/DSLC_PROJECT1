#!/usr/bin/env python
# coding: utf-8

# # DATA PREPARATION

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv('application_train.csv')
data1=pd.read_csv('application_test.csv')
data.head(10)


# In[3]:


data.shape#taille de mon dataset


# In[4]:


data['TARGET'].unique()


# In[5]:


def drop_60(df):
    nb_na=df.isna().sum()/df.shape[0]
    v=nb_na[nb_na>0.65]
    l=[i for i in v.index]
    data1=df.drop(l,axis=1)
    return data1


# In[6]:


data=drop_60(data)
data1=drop_60(data1)


# In[7]:


data.shape


# In[8]:


data.describe()


# In[9]:


data.dtypes.value_counts()#The differents types in my dataframes


# In[10]:


def check_na(feature):
    count=0
    list=[]
    for el in feature:
        if data[el].isna().sum()>0:
            list.append(el)
            count+=1
    print(data[list].isna().sum())
    print("{} columns with missing values".format(count))


# In[11]:


categorical=[var for var in data.columns if data[var].dtypes=='object']
categorical1=[var for var in data1.columns if data1[var].dtypes=='object']
categorical


# In[12]:


data.head(5)


# In[13]:


float_=[var for var in data.columns if data[var].dtypes=='float64']
float1_=[var for var in data1.columns if data1[var].dtypes=='float64']


# In[14]:


int_=[var for var in data.columns if data[var].dtypes=='int64']
int1_=[var for var in data1.columns if data1[var].dtypes=='int64']
int_


# In[15]:


data['DAYS_BIRTH'].unique()


# In[16]:


data['Age']=data['DAYS_BIRTH']/(-365)
data1['Age']=data1['DAYS_BIRTH']/(-365)
data['Age'].value_counts()


# In[17]:


data['DAYS_EMPLOYED'].unique()


# In[18]:


data['YEAR_EMPLOYED']=data['DAYS_EMPLOYED']/(-365)
data1['YEAR_EMPLOYED']=data1['DAYS_EMPLOYED']/(-365)
data['YEAR_EMPLOYED'].value_counts()


# In[19]:


#Number of missing value in categorical columns
check_na(categorical)


# In[20]:


check_na(categorical1)


# In[21]:


data['NAME_TYPE_SUITE'].unique()


# In[22]:


data[categorical]=data[categorical].fillna("Unknown"+data[categorical])
data1[categorical1]=data1[categorical1].fillna("Unknown"+data1[categorical1])


# In[23]:


check_na(categorical)


# In[24]:


check_na(float_)


# In[25]:


data[float_]= data[float_].fillna(data[float_].mean())
data1[float1_]= data1[float1_].fillna(data1[float1_].mean())


# In[26]:


check_na(int_)


# In[27]:


data=data.drop(['DAYS_BIRTH','DAYS_EMPLOYED'],axis=1)
data1=data1.drop(['DAYS_BIRTH','DAYS_EMPLOYED'],axis=1)


# In[28]:


data.head()


# In[29]:


data.shape


# In[30]:


cmap=abs(data.corr()['TARGET']).tail(10)
cmap


# In[31]:


data.head()


# In[32]:


sns.set(style="darkgrid")
x=data['TARGET'].value_counts()
sns.barplot(x.index,x.values,alpha=1)
plt.title('THE DISTRIBUTION OF REPAID LOAN OR NOT')
plt.xlabel('TARGET')
plt.ylabel('Number of observations')#most of the make loan have been repaid


# In[33]:


import plotly.express as px

fig = px.histogram(data['OCCUPATION_TYPE'], x="OCCUPATION_TYPE")
fig.update_layout(title_text='Number by occupation type clients')
fig.show()


# In[34]:


data['NAME_INCOME_TYPE'].unique()


# In[35]:


x=data[data['TARGET']==0]#on essaie d'en apprendre d'avantages sur les personnes qui ont eu à rembourser leurs prets


# In[36]:


target_0=pd.DataFrame(x)
target_0.head()

Le crédit renouvelable (revolving loan) est un contrat de prêt permettant de mettre à disposition de l'emprunteur une somme d'argent. Ce crédit est un contrat d'un an renouvelable. Le montant mis à la disposition de l'emprunteur se reconstitue au fur et à mesure de ses remboursements.

un crédit de trésorerie(cash loan) est un crédit à court terme qui peut-être aussi bien accordé aux entreprises qu'aux particuliers. Lorsqu'une entreprise y a recourt, celle-ci pourra disposer temporairement de la trésorerie utile à son fonctionnement. Le remboursement se fera bien entendu à court terme
# In[37]:


x=target_0['NAME_CONTRACT_TYPE'].value_counts()
sns.barplot(x.index,x.values,alpha=0.9)
plt.title('REPARTITION OF NAME CONTRACT TYPE BASED ON THE REPAID LOAN')
plt.xlabel('NAME CONTRACT TYPE')
plt.ylabel('Number of observations')#the majority of person who repaid their loan have done a cash loan(pret de tresorerie).it means that they already had enough money for their operations 


# In[38]:


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


# In[39]:


target_0['CNT_FAM_MEMBERS'].value_counts()


# In[40]:


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


# In[41]:


x=target_1['NAME_INCOME_TYPE'].head(1000).value_counts()
sns.barplot(x.index,x.values,alpha=0.9)
plt.title('REPARTITION OF NAME INCOME TYPE WHO REPAID LOAN WITH CASH LOAN')
plt.xlabel('NAME INCOME TYPE')
plt.ylabel('Number of observations')


# In[42]:


plt.figure(figsize=(10,5))#Age of persons who repairs their loan
plt.hist(target_1['Age'] , edgecolor = 'orange', bins = 20)
plt.title('Age of Client')
plt.xlabel('Age (years)')
plt.ylabel('Count')


# In[43]:


plt.figure(figsize=(10,5))
plt.xlim(data['AMT_CREDIT'].min(),data['AMT_CREDIT'].max())
plt.xlabel('AMT_CREDIT')
plt.ylabel('Density')
sns.kdeplot(data['AMT_CREDIT'],shade=True)
plt.title("Distribution of AMT_CREDIT")
plt.show()


# In[44]:


x=target_1['CODE_GENDER'].value_counts()
sns.barplot(x.index,x.values,alpha=0.9)
plt.title('REPARTITION OF GENDER WHO REPAID LOAN WITH CASH LOAN')
plt.xlabel('CODE GENDER')
plt.ylabel('Number of observations')#les femmes gagent lus que les hommes


# In[45]:


#les familles moins nombreuses(avec le moins d'enfants) ont donc plus de chance de rembourser leur credit
x=target_1['CNT_FAM_MEMBERS'].value_counts().head(6)
pie, ax = plt.subplots(figsize=[20,10])

plt.pie(x, autopct="%.1f%%", labels=x.index, pctdistance=0.5)
plt.title(" REPAID CASH LOAN BASED ON THE FAMILY NUMBER ", fontsize=15);
pie.savefig("DeliveryPieChart.png")

