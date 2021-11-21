import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
import os
import sys
from mlflow.utils.environment import _mlflow_conda_env
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
        
    data=pd.read_csv("mlflow_train.csv")
    
    X=data.loc[:, data.columns != 'TARGET']
    Y=data['TARGET']
    X=X.drop(['SK_ID_CURR'],axis=1)
    
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    alpha = int(sys.argv[1])
    with mlflow.start_run(run_name='random forest'):
        
        
        RF = RandomForestClassifier(n_estimators = alpha, random_state = 50, verbose = 1, n_jobs = -1,class_weight="balanced")
        RF.fit(X_train,Y_train)
      
        
      
        
        y_pred_auc = RF.predict_proba(X_test)[:,1]
        y_pred = RF.predict(X_test)
        
      
       
       
        roc_auc = roc_auc_score(Y_test, y_pred_auc)*100
        acc = accuracy_score(Y_test,y_pred)
        
        
        mlflow.log_metric("auc_roc",roc_auc)
        mlflow.log_metric("accuracy_score",acc)
        
       
        
       
        mlflow.log_param('n_estimators', alpha)
        mlflow.log_param('random_state', 50)
        mlflow.log_param('verbose', 1)
        mlflow.log_param('n_jobs', -1)
        mlflow.log_param('class_weight','balanced')
        


        # log model
        mlflow.sklearn.log_model(RF, "model")
        print("roc_auc",roc_auc)
        print("accuracy_score",acc)
