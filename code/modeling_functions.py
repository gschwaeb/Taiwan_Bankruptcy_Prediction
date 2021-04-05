import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost
import time
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score, f1_score, log_loss, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks 
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.dummy import DummyClassifier

def clean_data(df):
    
    df.columns = [x[1:] for x in df.columns]
    df = df.rename({'ankrupt?': 'Bankrupt?'}, axis=1)
    df = df[['Bankrupt?','ROA(C) before interest and depreciation before interest','ROA(A) before interest and % after tax','ROA(B) before interest and depreciation after tax','Operating Gross Margin','Realized Sales Gross Margin','Operating Profit Rate','Pre-tax net Interest Rate','Non-industry income and expenditure/revenue','Operating Expense Rate','Cash flow rate','Interest-bearing debt interest rate','Realized Sales Gross Profit Growth Rate','Operating Profit Growth Rate','Regular Net Profit Growth Rate','Continuous Net Profit Growth Rate','Total Asset Growth Rate','Current Ratio','Quick Ratio','Interest Expense Ratio','Total debt/Total net worth','Debt ratio %','Total Asset Turnover','Inventory Turnover Rate (times)','Fixed Assets Turnover Frequency','Working Capital to Total Assets','Quick Assets/Total Assets','Current Assets/Total Assets','Cash/Total Assets','Quick Assets/Current Liability','Cash/Current Liability','Current Liability to Assets','Operating Funds to Liability','Inventory/Working Capital','Inventory/Current Liability','Current Liabilities/Liability','Current Liabilities/Equity','Long-term Liability to Current Assets','Retained Earnings to Total Assets','Total expense/Assets','Current Asset Turnover Rate','Quick Asset Turnover Rate','Working capitcal Turnover Rate','Fixed Assets to Assets','Current Liability to Liability','Current Liability to Equity','Cash Flow to Total Assets','Cash Flow to Liability','CFO to Assets','Cash Flow to Equity','Current Liability to Current Assets','Liability-Assets Flag','Net Income to Total Assets','No-credit Interval','Gross Profit to Sales','Liability to Equity','Degree of Financial Leverage (DFL)','Interest Coverage Ratio (Interest expense to EBIT)','Net Income Flag','Equity to Liability', 'Allocation rate per person']]

    df.drop('Net Income Flag', axis = 1, inplace=True)

    original_categorical_cols = ['Liability-Assets Flag']
    original_numerical_cols = ['ROA(C) before interest and depreciation before interest','ROA(A) before interest and % after tax','ROA(B) before interest and depreciation after tax','Operating Gross Margin','Realized Sales Gross Margin','Operating Profit Rate','Pre-tax net Interest Rate','Non-industry income and expenditure/revenue','Operating Expense Rate','Cash flow rate','Interest-bearing debt interest rate','Realized Sales Gross Profit Growth Rate','Operating Profit Growth Rate','Regular Net Profit Growth Rate','Continuous Net Profit Growth Rate','Total Asset Growth Rate','Current Ratio','Quick Ratio','Interest Expense Ratio','Total debt/Total net worth','Debt ratio %','Total Asset Turnover','Inventory Turnover Rate (times)','Fixed Assets Turnover Frequency','Working Capital to Total Assets','Quick Assets/Total Assets','Current Assets/Total Assets','Cash/Total Assets','Quick Assets/Current Liability','Cash/Current Liability','Current Liability to Assets','Operating Funds to Liability','Inventory/Working Capital','Inventory/Current Liability','Current Liabilities/Liability','Current Liabilities/Equity','Long-term Liability to Current Assets','Retained Earnings to Total Assets','Total expense/Assets','Current Asset Turnover Rate','Quick Asset Turnover Rate','Working capitcal Turnover Rate','Fixed Assets to Assets','Current Liability to Liability','Current Liability to Equity','Cash Flow to Total Assets','Cash Flow to Liability','CFO to Assets','Cash Flow to Equity','Current Liability to Current Assets','Net Income to Total Assets','No-credit Interval','Gross Profit to Sales','Liability to Equity','Degree of Financial Leverage (DFL)','Interest Coverage Ratio (Interest expense to EBIT)','Equity to Liability']
    cols_to_check_zero = ['ROA(C) before interest and depreciation before interest','ROA(A) before interest and % after tax','ROA(B) before interest and depreciation after tax','Operating Gross Margin','Realized Sales Gross Margin','Operating Profit Rate','Pre-tax net Interest Rate','Non-industry income and expenditure/revenue','Operating Expense Rate','Cash flow rate','Realized Sales Gross Profit Growth Rate','Operating Profit Growth Rate','Regular Net Profit Growth Rate','Continuous Net Profit Growth Rate','Total Asset Growth Rate','Current Ratio','Quick Ratio','Interest Expense Ratio','Total debt/Total net worth','Debt ratio %','Total Asset Turnover','Inventory Turnover Rate (times)','Fixed Assets Turnover Frequency','Working Capital to Total Assets','Quick Assets/Total Assets','Current Assets/Total Assets','Cash/Total Assets','Quick Assets/Current Liability','Cash/Current Liability','Current Liability to Assets','Operating Funds to Liability','Inventory/Working Capital','Current Liabilities/Liability','Current Liabilities/Equity','Retained Earnings to Total Assets','Total expense/Assets','Current Asset Turnover Rate','Quick Asset Turnover Rate','Working capitcal Turnover Rate','Fixed Assets to Assets','Current Liability to Liability','Current Liability to Equity','Cash Flow to Total Assets','Cash Flow to Liability','CFO to Assets','Cash Flow to Equity','Current Liability to Current Assets','Net Income to Total Assets','No-credit Interval','Gross Profit to Sales','Liability to Equity','Degree of Financial Leverage (DFL)','Interest Coverage Ratio (Interest expense to EBIT)','Equity to Liability']

    df = df.loc[(df[cols_to_check_zero] != 0).all(axis=1)]

    return df

def perform_eda_analysis(df):
    
    fig, axes = plt.subplots(2, 3, figsize=(24,12))

    sns.distplot(df[df['Bankrupt?']==0]['ROA(A) before interest and % after tax'].apply(lambda x: x*100), color = 'green', label = 'Not Bankrupt', ax=axes[0,0])
    sns.distplot(df[df['Bankrupt?']==1]['ROA(A) before interest and % after tax'].apply(lambda x: x*100), color = 'red', label = 'Bankrupt', ax=axes[0,0])
    axes[0,0].set_title('Distributions of ROA(A) for Bankrupt and Not Bankrupt Companies', fontsize = 15)
    axes[0,0].legend(loc = 'upper right')

    sns.distplot(df[df['Bankrupt?']==0]['Debt ratio %'].apply(lambda x: x*100), color = 'green', label = 'Not Bankrupt', ax=axes[0,1])
    sns.distplot(df[df['Bankrupt?']==1]['Debt ratio %'].apply(lambda x: x*100), color = 'red', label = 'Bankrupt', ax=axes[0,1])
    axes[0,1].set_title('Debt ratio % for Bankrupt and Not Bankrupt Companies', fontsize = 15)
    axes[0,1].legend(loc = 'upper right')

    sns.distplot(df[(df['Bankrupt?']==0) & (df['Quick Ratio'] <1)]['Quick Ratio'].apply(lambda x: x*100), color = 'green', label = 'Not Bankrupt', ax=axes[0,2])
    sns.distplot(df[(df['Bankrupt?']==1) & (df['Quick Ratio'] <1)]['Quick Ratio'].apply(lambda x: x*100), color = 'red', label = 'Bankrupt', ax=axes[0,2])
    axes[0,2].set_title('Quick Ratio Bankrupt and Not Bankrupt Companies', fontsize = 15)
    axes[0,2].legend(loc = 'upper right')

    sns.distplot(df[df['Bankrupt?']==0]['Current Liability to Assets'].apply(lambda x: x*100), color = 'green', label = 'Not Bankrupt', ax=axes[1,0])
    sns.distplot(df[df['Bankrupt?']==1]['Current Liability to Assets'].apply(lambda x: x*100), color = 'red', label = 'Bankrupt', ax=axes[1,0])
    axes[1,0].set_title('Current Liability to Assets for Bankrupt and Not Bankrupt Companies', fontsize = 15)
    axes[1,0].legend(loc = 'upper right')

    sns.distplot(df[df['Bankrupt?']==0]['Working Capital to Total Assets'].apply(lambda x: x*100), color = 'green', label = 'Not Bankrupt', ax=axes[1,1])
    sns.distplot(df[df['Bankrupt?']==1]['Working Capital to Total Assets'].apply(lambda x: x*100), color = 'red', label = 'Bankrupt', ax=axes[1,1])
    axes[1,1].set_title('Distributions of Working Capital to Total Assets', fontsize = 15)
    axes[1,1].legend(loc = 'upper right')

    sns.boxplot(x="Bankrupt?", y='Long-term Liability to Current Assets', data=df[df['Long-term Liability to Current Assets']>1],ax=axes[1,2]) 
    axes[1,2].set_title('Long-term Liability to Current Assets When Ratio Exceeds 1', fontsize=14)
    axes[1,2].get_yaxis().get_major_formatter().set_scientific(False)

    fig.tight_layout()

    fig2, axes = plt.subplots(1, 4, figsize=(12,3))
    for xcol, ax in zip(['Total Asset Turnover', 'Current Liability to Current Assets', 'Degree of Financial Leverage (DFL)', 'Debt ratio %'], axes):
        df.plot(kind='scatter', x=xcol, y='Bankrupt?', ax=ax, alpha=0.4, color='b', yticks=[], xticks=[])

    fig2.tight_layout()

    sns.pairplot(df, vars = ['ROA(C) before interest and depreciation before interest', 'Liability to Equity', 'Current Ratio', 'Interest Coverage Ratio (Interest expense to EBIT)', 'Working Capital to Total Assets'], hue = 'Bankrupt?', diag_kind = 'kde', plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, height = 4);
    
    
def perform_feature_engineering(df):
    
    df['Good_CurrBS_Standing'] = np.where(df['Current Liability to Current Assets'] > 0.3, 1, 0)
    df['Volatile_Earnings'] = np.where(df['Degree of Financial Leverage (DFL)'] > 0.1, 1, 0)
    df['High_Asset_Turnover'] = np.where(df['Total Asset Turnover'] > 0.5, 1, 0)
    df['Healthy_Company'] = np.where(df['ROA(C) before interest and depreciation before interest'] > 0.25, 1, 0)
    df['Continuous Net Profit Growth Rate'] = np.where(df['Continuous Net Profit Growth Rate']>3, 3, df['Continuous Net Profit Growth Rate'])

    df['Good_Quick_Ratio'] = np.where(df['Quick Ratio'] > .005, 1, 0)
    df['Good_Current_Ratio'] = np.where(df['Current Ratio'] > .01, 1, 0)
    df['Good_Debt_toNetWorth'] = np.where(df['Total debt/Total net worth'] < .012, 1, 0)
    df['Good_Inventory_Turnover'] = np.where(df['Inventory Turnover Rate (times)'] >= 1, 1,0)

    df['Good_Cash_toLiability'] = np.where(df['Cash/Current Liability'] >1, 1, 0)
    df['Total_Assets_Growing?'] = np.where(df['Total Asset Growth Rate'] >=1, 1, 0)
    df['Good_Operating_Expense_Rate'] = np.where(df['Operating Expense Rate'] <1, 1, 0)
    df['Inventory_Excess_Over_Current_Liability'] = np.where(df['Inventory/Current Liability'] >=1,1,0)
    df['LTL_Excess_to_CA'] = np.where(df['Long-term Liability to Current Assets'] >= 1, 1, 0)
    df['Excess_Sales_to_CA'] = np.where(df['Current Asset Turnover Rate'] >=1,1,0)
    df['Excess_Sales_to_Quick_Assets'] = np.where(df['Quick Asset Turnover Rate'] >=1,1,0)

    df['Allocation rate per person'] = np.where(df['Allocation rate per person'] >1,.8,df['Allocation rate per person'])
    df['Quick Assets/Current Liability'] = np.where(df['Quick Assets/Current Liability'] >1, 0.275, df['Quick Assets/Current Liability'])

    return df

def check_correlations(df):
    sns.set(style="white")
    data = df[best_selectk10_features].corr()
    mask = np.zeros_like(data, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(60,50))
    cmap = sns.diverging_palette(133, 10, as_cmap=True)  
    g = sns.heatmap(data=data, annot=True, cmap=cmap, ax=ax, mask=mask, annot_kws={"size":20}, cbar_kws={"shrink": 0.8} )
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 40, rotation =90)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 40)


def get_best_features(xdata, yvals, num_features):

    best50_rf_features = ['ROA(C) before interest and depreciation before interest','ROA(A) before interest and % after tax','ROA(B) before interest and depreciation after tax','Operating Gross Margin','Realized Sales Gross Margin','Operating Profit Rate','Pre-tax net Interest Rate','Non-industry income and expenditure/revenue','Cash flow rate','Realized Sales Gross Profit Growth Rate','Operating Profit Growth Rate','Regular Net Profit Growth Rate','Continuous Net Profit Growth Rate','Total Asset Growth Rate','Current Ratio','Interest Expense Ratio','Debt ratio %','Total Asset Turnover','Working Capital to Total Assets','Quick Assets/Total Assets','Current Assets/Total Assets','Cash/Total Assets','Quick Assets/Current Liability','Current Liability to Assets','Operating Funds to Liability','Inventory/Working Capital','Current Liabilities/Liability','Current Liabilities/Equity','Retained Earnings to Total Assets','Total expense/Assets','Working capitcal Turnover Rate','Fixed Assets to Assets','Current Liability to Liability','Current Liability to Equity','Cash Flow to Total Assets','Cash Flow to Liability','CFO to Assets','Cash Flow to Equity','Current Liability to Current Assets','Net Income to Total Assets','No-credit Interval','Gross Profit to Sales','Liability to Equity','Degree of Financial Leverage (DFL)','Interest Coverage Ratio (Interest expense to EBIT)','Equity to Liability','Good_Quick_Ratio_1','Good_Debt_toNetWorth_1']

    selector = SelectKBest(k=num_features)
    selector.fit(xdata, yvals) 
    best_selectk50_features = list(xdata.columns[selector.get_support()])

    best50_rf_features.extend(best_selectk50_features)
    all_best_features = list(set(best50_rf_features))

    return all_best_features

def return_best_model(input_model, params, xtrain, ytrain, resampling_method = []):
    
    kf = KFold(n_splits=5, shuffle=False)

    model_pipeline = make_pipeline(StandardScaler(), *resampling_method, input_model)
    grid_pipeline = GridSearchCV(model_pipeline, param_grid=params, cv=kf, scoring='recall', verbose=0)
    grid_pipeline.fit(xtrain, ytrain)
    print(grid_pipeline.best_params_)

    return grid_pipeline

def best_model_score(pipeline_instance):    
    return { 'best_params' : pipeline_instance.best_params_, 'cv_mean_recall' : pipeline_instance.best_score_, 'pipeline_instance' : pipeline_instance, 'best_estimator' : pipeline_instance.best_estimator_ }

def get_model_params(classifier):
    
    if (classifier == 'logisticregression'):
    
        model_instance = LogisticRegression(max_iter=10000)
        hyper_params = {'logisticregression__solver' : ['liblinear'],
                        'logisticregression__penalty': ['l1'],
                        'logisticregression__C': [0.01, 0.1, 10],
                        'logisticregression__class_weight': ['balanced'] }

    elif (classifier == 'kneighbors'):
    
        model_instance = KNeighborsClassifier()
        hyper_params = {'kneighborsclassifier__n_neighbors': [3, 9,17],
                        'kneighborsclassifier__weights': ['uniform','distance']
                         }

    elif (classifier == 'decisiontree'):
    
        model_instance = DecisionTreeClassifier()
        hyper_params = {'decisiontreeclassifier__criterion': ['entropy','gini'],
                        'decisiontreeclassifier__splitter': ['best','random'],
                        'decisiontreeclassifier__max_depth': [None, 5, 10],
                        'decisiontreeclassifier__min_samples_split': [3, 5, 10], 
                        'decisiontreeclassifier__class_weight': [None, 'balanced'] }
    
    elif (classifier == 'randomforest'):
    
        model_instance = RandomForestClassifier()
        hyper_params = {'randomforestclassifier__max_features': ['auto','sqrt'],
                        'randomforestclassifier__bootstrap': [True, False] }

    elif (classifier == 'bagging'):
    
        model_instance = BaggingClassifier()
        hyper_params = {'baggingclassifier__base_estimator': [DecisionTreeClassifier(), KNeighborsClassifier],
                          'baggingclassifier__max_features': [0.8, 0.5],
                          'baggingclassifier__max_samples': [0.8, 0.5] }

    elif (classifier == 'gradientboosting'):
    
        model_instance = GradientBoostingClassifier()
        hyper_params = {'gradientboostingclassifier__loss': ['deviance', 'exponential'],
                        'gradientboostingclassifier__min_samples_split': [3, 5],
                        'gradientboostingclassifier__max_depth': [3, 10] }

    return model_instance, hyper_params

