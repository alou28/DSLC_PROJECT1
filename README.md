# DSLC_PROJECT1

## APPLICATION OF BIG DATA

This project is for the course "Application of Big Data" By:

* Alain NGOMEDJ
* Emeric BERTIN
* BACHACHA Hassan


## Introduction:

The goal of this project is to apply some concepts & tools to multiple part of the project:
For this project we will be using the main data of dataset of home and credit risk classification from kaggle:
                                  https://www.kaggle.com/c/home-credit-default-risk

It contains various informations about previous loans and the if the loans has been repayed by the borrower.

## Goals :

* Part 1: Buil a Classical ML projects with respect to basic ML Coding best practices
* Part 2: Integrate MLFlow to our project
* Part 3: Integrate ML Interpretability to our project which is about shap


## Project Organization

* __DSLC_PROJECT__ : the root directory of the project
  * __Data__ : the directory containing all the data used for the project
    * __application_train.csv__ :Data from the dataset in the kaggle The initial dataset, its contain information about loans and loans applicants (at application time), each line is a unique application
    * __application_test.csv__ : The initial dataset, its contain information about loans and loans applicants (at application time), each line is a unique application
    * __mlflow_train.csv__ : The initial dataset, its contain information about loans and loans applicants (at application time), each line is a unique application
* __Model__ : the directory containing all the models trained for that project saved as pickled files
  * __GradientBoosting.pkl__ : A gradient boosting model
  * __RandomForest.pkl__ : A random forest model
  * __XGBoost.pkl__ : An Extra Gradient Boosting model
* __Notebook__ : The directory containing the 6 notebooks used for that project
  * __Data_preparation.ipynb__ : A notebook which collect the data from application_train.csv come from a [Kaggle contest](.csv and create dataset_prepared.csv
  * __Features_engineering.ipynb__ : A notebook that collects the data from dataset_prepared.csv and creates dataset_final.csv
  * __Model_training.ipynb__: A notebook that splits the data from dataset_final.csv into a test and a training dataset, then used the training dataset to train the 3 models
  * __MLFLOW_PART.ipynb__: A notebook used to evaluate the trained models
* __mlruns__ : A directory created when using MLflow which contains the runs' logs
* __ReadMe.md__ : The file you are currently reading
* __.gitignore__ : A file that is used to exclude files that shan't be on the git either because they're not relevant (like Jupyter's logs) or because they're too big (like the models and datasets)

## Data Exploration

## Feature Engineering

The first part of the project is to build an ML Project with respect of using 3 machine learning models:
Random Forest
XGBoost
Gradient Boosting
But our goal isn't just to create three machine learning models. Indeed, these models' accuracies aren't very important. In fact the real goal is to implement multiple tools into the project, in order to make the project more understandable for everyone. The tools to be used are:

Git
Conda environment
Mlflow
SHAP
Structuration
After creating a Git repository and granting access to everyone that needed it, we decided to start computing models on Jupyter Lab in a Conda environment. We decided to not use multiple branches in the Git repository, as we thought it wasn't necessary. For the project documentation, we used Sphinx.

# Project Outputs

## Sphinx
![knlk](https://user-images.githubusercontent.com/93646318/143021301-7f030118-fdeb-4156-84c2-beb5ad34ac74.PNG)
![bhjb](https://user-images.githubusercontent.com/93646318/143021313-6b4eaf73-7eae-4b2e-ad16-a485e5097d7d.PNG)

## Mlflow ui
![kehirfb](https://user-images.githubusercontent.com/93646318/143021243-ef60f0bd-1a1e-4305-b516-4e39cea69106.PNG)
![jbfd](https://user-images.githubusercontent.com/93646318/143021350-d80148a7-3b5e-4341-baf7-82140cdcffa9.PNG)
## SHAP
![jbnj](https://user-images.githubusercontent.com/93646318/143021381-9e17e483-311a-4eb9-bc4a-89f37bcb8ec1.PNG)
![Capture](https://user-images.githubusercontent.com/93646318/143021388-f9b8ac9f-eaaf-48a6-b118-31e99bd818d2.PNG)


