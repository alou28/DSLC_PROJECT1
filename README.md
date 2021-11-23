# DSLC_PROJECT1 - APPLICATION OF BIG DATA -

This project is for the course "Application of Big Data", made by:

* Alain NGOMEDJ
* Emeric BERTIN
* BACHACHA Hassan


## Introduction:

The goal of this project is to apply some concepts & tools to multiple part of the project:
For this project we will be using the main data of dataset of home and credit risk classification from kaggle:
                                  https://www.kaggle.com/c/home-credit-default-risk

It contains various informations about previous loans and the if the loans has been repayed by the borrower.

## Goals :

* Part 1: Build a Classical ML projects with respect to basic ML Coding best practices
* Part 2: Integrate MLFlow to our project
* Part 3: Integrate ML Interpretability to our project which is about shap


## Project Organization

* __DSLC_PROJECT__ : the root directory of the project
  * __application_train.csv__ :Data from the dataset in the kaggle The initial dataset, its contain information about loans and loans applicants (at application time), each line is a unique application
  * __application_test.csv__ : The initial dataset, its contain information about loans and loans applicants (at application time), each line is a unique application that we need to predict at the ned 
* __mlflow_train.csv__ : The dataset used to train the model with mlflow.this dataset is already cleaned and standardized
* __Notebook__ : The directory containing the 6 notebooks used for that project
  * __DDSLC_PROJECT__ : The notebook with all the informations
  * __Data_preparation.ipynb__ : A notebook which collect the data from application_train.csv and application_test.csv
  * __Features_engineering.ipynb__ : A notebook that encode the categorical columns values and then split the data into train and test values. 
  * __Model_training and prediction.ipynb__: A notebook to train the 3 machine learning models
* __mlruns__ : A directory created when using MLflow which contains the runs' logs
* __train.py__: A py file used to deploy the model with mlflow 
* __ReadMe.md__ : The file you are currently reading
* __.gitignore__ : A file that is used to exclude files that shan't be on the git either because they're not relevant (like Jupyter's logs) or because they're too big (like the models and datasets)

## Data Exploration and Data cleaning

For the data exploration we focus on Target data to see the importance of this column into the dataset.

![krkr](https://user-images.githubusercontent.com/93646318/143035504-4fd45f32-fe80-408f-938f-7001642b46c0.PNG)

![1](https://user-images.githubusercontent.com/93646318/143035522-9b91be5b-9852-437a-9a29-538f3729efb9.PNG)

![2](https://user-images.githubusercontent.com/93646318/143035551-f8b924f8-f893-48c0-95cd-e5221c59d488.PNG)


*Data cleaning

First, we drop columns with more than 65% missing values. Then, we replace "DAYS_BIRTH" and "DAYS_EMPLOYED" by numerical values.

## Feature Engineering

The first part of the project is to build an ML Project with respect of using 3 machine learning models:
* Random Forest
* XGBoost
* Gradient Boosting
But our goal isn't just to create three machine learning models. 

Indeed, these models' accuracies aren't very important. 
The real goal is to implement multiple tools into the project, in order to make the project more understandable for everyone.
to do that we will be using this precise workflow :

* Conda environment
* Git: Creating a Git repository and granting access to everyone
* Sphinx: Documentation Library 
* Mlflow: Help manage the complete machine learning lifecycle of a data science Project
* SHAP: Game theoretic approach to explain the output of any machine learning model.

First, We encode the categorical columns values and then split the data into train and test values. 
Now, our datas are ready for the model training and prediction.

# Models Training and prediction

## XGBOOST

![image](https://user-images.githubusercontent.com/93646318/143036266-eca2d339-cd3c-41a0-b6b0-16fde59b9d6a.png)

![image](https://user-images.githubusercontent.com/93646318/143036340-31c15343-c866-4b95-8b39-1f2c321efaf3.png)

![image](https://user-images.githubusercontent.com/93646318/143036409-f9221052-a2b6-4a69-a064-1425fb11b719.png)

## Random Forest

![image](https://user-images.githubusercontent.com/93646318/143036474-1b46be64-5b90-481e-aac9-631854746ea2.png)

![image](https://user-images.githubusercontent.com/93646318/143036504-293f8c72-11d6-4a3a-a4cf-89a0c1d4ef68.png)

## Gradient Boost

![image](https://user-images.githubusercontent.com/93646318/143036593-22324be5-5f98-4b1d-bef9-8e6e568e8e40.png)

# Project Outputs

## Sphinx
To access to sphinx in the path repesitory in the terlinal, we need to type:
1. make html
2. cd _build\html
3. index.html

![knlk](https://user-images.githubusercontent.com/93646318/143021301-7f030118-fdeb-4156-84c2-beb5ad34ac74.PNG)
![bhjb](https://user-images.githubusercontent.com/93646318/143021313-6b4eaf73-7eae-4b2e-ad16-a485e5097d7d.PNG)

## Mlflow ui
To access to mlflow in the path repesitory, we just need to type 

* _mlflow ui_

To access the ui we just need to copy and paste the local url into a web browser.

By running Mlflow ui from our base folder, it creates a folder named mlruns that contains our experiments run with all the information about the model.

![ke,rl](https://user-images.githubusercontent.com/93646318/143025579-db0e6e0e-6f3b-4f29-b4a5-f3716878eacb.PNG)

We can see Mlflow running, and access to all the runs 

![kehirfb](https://user-images.githubusercontent.com/93646318/143021243-ef60f0bd-1a1e-4305-b516-4e39cea69106.PNG)

![jbfd](https://user-images.githubusercontent.com/93646318/143021350-d80148a7-3b5e-4341-baf7-82140cdcffa9.PNG)

![kdjd](https://user-images.githubusercontent.com/93646318/143025810-d29e4d92-9aba-43ea-8fe9-0ed1de2f3839.PNG)

Deployment of packaged model 
We used Random Forest to train our mlflow model.
* mlflow run . -P alpha=200
* mlflow ui

Deployment the model in the local REST, using the command line  :

mlflow models serve --model-uri mlruns\0\run-id\artifacts\model -p
 1234


![hdhjfhdf](https://user-images.githubusercontent.com/93646318/143030869-4868e5e2-0308-4864-85ef-613c24b51589.PNG)

![kekdk](https://user-images.githubusercontent.com/93646318/143030925-a54b44ec-aefa-4a5e-8bd7-f1ff8219aa28.PNG)

## SHAP

SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.

![image](https://user-images.githubusercontent.com/93646318/143036701-5ed657ff-0406-4a0e-9607-a5a0da982877.png)

![shap](https://user-images.githubusercontent.com/93646318/143031053-bd68d9e3-0a52-46bc-af2b-d9dc72517b8f.PNG)

![jbnj](https://user-images.githubusercontent.com/93646318/143021381-9e17e483-311a-4eb9-bc4a-89f37bcb8ec1.PNG)



