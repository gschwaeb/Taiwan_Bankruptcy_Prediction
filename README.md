# Taiwan Bankruptcy Prediction
#### by Gary Schwaeber and Sunil Rao

## Overview - Business Problem
The business problem we are trying to solve is to help bank lenders by creating a predictive model for which companies will go bankrupt. 

## Data

The dataset, which was downloaded from Kaggle, contains financial ratios and a bankrupcty indicator for approximately 7000 Taiwanese companies. The data, which was originally gathered for a [research report](https://isslab.csie.ncu.edu.tw/download/publications/1.pdf) and was taken from the Taiwan Economic
Journal for the years 1999–2009. 

## Exploratory Data Analysis & Feature Engineering

First, we began by cleaning the data, filtering down the columns from 100 to 67 and resolved the class imbalance issue presented in the data. 

Next, We focused our efforts on the most critical metrics related to a company's ability to pay interest on debt owed, specifically metrics related to Liabilities, Cash Flow and Assets. We also used correlation matrices to better identify best predictors. 

## Building Tuning and Testing Models

We tested our hypothesis on the following models: logistic regression, K nearest neighbors, decision trees, random forest, bagging and gradient boosting to determine which algorithm performs best.

To handle the class imbalance we will test various models using SMOTE, Tomek Links, SMOTE and Tomek Links together, as well as no resampling. 

## Model Cross Validation Training Scores

From the cross validation results, it appears Logistic Regression generally performs best, followed by decision trees, then K nearest neighbors. The different resampling methods also seem to have affected each of the models differently.

## Evaluate Best Models to Test Data 

The best performing model, optimizing for recall, was logistic regression fit using Tomek Links resampling with paramters C = 0.01, class_weight = 'balanced', penalty = 'l1', solver = 'liblinear'. We believe that given similar market conditions to the years covered in our dataset and access to similar data, we could deploy this model to unseen data and continue to predict roughly 87% of companies that go bankrupt out of the companies that actually did go bankrupt.

### Next Steps
In order to improve our modelling the below further steps can be taken
- Add more company financial data outside of the given financial ratios
- Add company industries and industry specific data
- Add local and global economic condition data for the time period
- Test with more classification machine learning algorithms and hyperparamters for tuning the models we used. With limited time and computing power we had to cut many hyperparameters to test in our grid search

## Repository Structure

```
├── code
│   ├── __init__.py
│   ├── modeling_functions.py
├── .gitignore
├── Readme.md
├── data
├── Taiwan_Bankruptcy_Prediction_Final.ipynb
└── final_results.pickle
```
