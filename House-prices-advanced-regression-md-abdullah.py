#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: center;">
#     <img src="https://storage.googleapis.com/kaggle-media/competitions/House%20Prices/kaggle_5407_media_housesbanner.png" alt="Image" style="width: auto; height: auto;">
# </div>
# 
# <br/>
# <div style = "text-align: center"><font size = 6 color = "#B22222" face = "verdana"><b>House Prices - Advanced Regression Techniques</b></font></div> <br/> 
# <div style = "text-align: center"><font size = 5 color = "#00008B" face = "verdana"><b>Md.Abdullah-Al Mamun</b></font></div>

# <a id = "table-of-contents"></a>
# # Table of Contents   
#     
# [1. Introduction](#intro)
#    - [About the dataset](#intro)
#    - [Information about the dataset](#information)
#    - [Identify the features and the targets](#features)
#    - [Label Encoding Categorical Columns](#Encoding)
# 
# 
# 
# 
# 
# [2. Data Exploration and Analysis](#Exploration)
#    - [Visualizing the distribution ](#Exploration)
#    - [The correlation matrix](#correlation)
#    - [Data Cleaning,Checking errors, Missing values](#Cleaning)
#    - [Correlation matrix heatmap](#heatmap)
#    - [Outlier detection(Normality test)](#Outlier_detection)
#    - [Visualizing outliers](#Outlier_viz)
#    
# 
# 
# 
# 
# [3. Feature Engineering](#Feature_Engineering)
#    - [Checking if transformation is needed](#Feature_Engineering)
#    - [Transformation](#Transformation)
#    - [Feature importance Analysis](#Feature_importance)
#    - [Feature Selection and visualization](#Feature_Selection)
#    - [Feature rankings](#Feature_rankings)
# 
# 
# 
# 
# [4. Model Building and Evaluation](#Model)
#    - [The best model and interpretation](#best_Model)
#    - [Model Evaluation and Comparison](#compare)
#    - [Hyperparameter Tuning and Model Selection](#Tuning)
#    - [Model Prediction using test data](#using_test_data)
# 
#    
#    
#   
#   
# [5. Deep Learning (Pytorch Lightening)](#Pytorch)
#    - [Visualizing the Losses(Train VS Validation Loss)](#Losses)
#    - [End Result](#Final_Prediction)
# 
#   
# 
# 

# <a id="intro"></a>
# 
# 
# # 1. Introduction
# 
# ## About the dataset
# 
# This dataset encompasses a comprehensive array of features detailing various aspects of residential properties in Ames, Iowa. It includes 79 explanatory variables, encompassing a wide range of information beyond just the number of bedrooms or the presence of a white-picket fence. These variables shed light on diverse aspects of the homes, allowing for a nuanced understanding of their characteristics.
# 
# ## Goal:
# 
# The primary objective of this task is to predict the final sale price for each house in the dataset. 
# 

# ### Libraries

# In[1]:


# Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import warnings
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


# ### Loading the training data

# In[2]:


# Loading the training data
data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
data 


# <a id="information"></a>
# 
# ### Getting information about the dataset

# In[3]:


# Getting information about the dataset.

info = data.info()
info


# ### Description of the data

# In[4]:


description = data.describe()
description 


# In[5]:


# Checking the missing values in each coulmn

missing_values = data.isnull().sum()
missing_values


# <a id="features"></a>
# 
# ## Identify the features and the targets:
# 
# 
# ### Targets/labels: 'SalePrice' column
# - The target variable for this project is 'SalePrice'.
# - The goal is to predict the final sale price for each house in the dataset.
# 
# 
# ### Features(predictor):  All columns except 'SalePrice' from the dataset.
# 
# - These are the explanatory variables that provide information about residential properties in Ames, Iowa.
# - They include columns such as 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', and many more, comprising a total of 79 explanatory variables.

# <a id="Encoding"></a>
# 
# ## Label Encoding Categorical Columns
# 

# In[6]:


# Encodes categorical columns using LabelEncoder.

def Label_Encoder(df):
    object_cols = df.select_dtypes(include='object').columns
    for col in object_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

data = Label_Encoder(data)
data


# <a id="Exploration"></a>
# # 2. Data Exploration and Analysis
# ### Visualizing the distribution
# 

# In[7]:


#Plotting the distribution of each column in the DataFrame.
    
def plot_distribution(data):
    num_columns = len(data.columns)
    num_plots_per_row = 4
    num_rows = -(-num_columns // num_plots_per_row) 

    plt.figure(figsize=(20, 5 * num_rows))  

    for i, column in enumerate(data.columns, 1):
        plt.subplot(num_rows, num_plots_per_row, i)
        sns.histplot(data[column], kde=True)
        plt.title(f"Distribution of {column}", fontsize=14)  
        plt.xlabel(column, fontsize=12)  
        plt.ylabel('Frequency', fontsize=12)  

        if i % num_plots_per_row == 0 or i == num_columns:
            plt.subplots_adjust(wspace=0.5, hspace=0.5) 

    plt.tight_layout()  # Adjust layout
    plt.show()


plot_distribution(data)


# <a id="correlation"></a>
# ## The correlation matrix

# In[8]:


# # Calculating the absolute correlation matrix

correlation_matrix = abs(data.corr()).round(2)
correlation_matrix


# ### Checking the correlations with SalePrice

# In[9]:


# Getting the absolute correlations with SalePrice, sorted in descending order

target_correlations = correlation_matrix['SalePrice'].abs().sort_values(ascending=False)
target_correlations 


# <a id="Cleaning"></a>
# 
# ## Data Cleaning
# ### Checking errors 

# ### Used my own module:
# 
# import pandas as pd
# 
# class DataFrameChecker:
#     def __init__(self, data):
#         self.data = data
# 
#     def check_errors(self):
#         has_errors = False
#         print("Errors in DataFrame:")
#         for column in self.data.columns:
#             try:
#                 # Check for missing values in the column
#                 if self.data[column].isnull().any():
#                     has_errors = True
#                     print("Error: Missing values found in column:", column)
#                     print(self.data[self.data[column].isnull()])
# 
#                 # Check for out-of-range errors
#                 errors = self.data[(self.data[column] < self.data[column].min()) | (self.data[column] > self.data[column].max())]
#                 if not errors.empty:
#                     has_errors = True
#                     print("Errors found in column:", column)
#                     print(errors)
# 
#             except TypeError:
#                 print("Error: Non-numeric values found in column:", column)
#                 has_errors = True
# 
#         if not has_errors:
#             print("No errors found in DataFrame")
# 
#     def check_duplicates(self):
#         duplicates = self.data[self.data.duplicated()]
#         if not duplicates.empty:
#             print("Duplicates found in DataFrame:")
#             print(duplicates)
#         else:
#             print("No duplicates found in DataFrame")
# 
#     def check_missing_values(self):
#         missing_values = self.data.isnull().sum()
#         if missing_values.sum() > 0:
#             print("Missing values found in DataFrame:")
#             print(missing_values[missing_values > 0])
#         else:
#             print("No missing values found in DataFrame")

# In[11]:


from DataFrame_Checker import DataFrameChecker

# An instance of DataFrameChecker
checker = DataFrameChecker(data)

# Called the checking functions
checker.check_errors()


# ### checking missing values

# In[12]:


# checking missing values
checker.check_missing_values()


# ### Dropping rows with missing values

# In[13]:


# Dropping rows with missing values
data.dropna(axis=0, inplace=True)


# ### Checking after dropping the rows with missing values

# In[14]:


checker.check_missing_values()


# <a id="heatmap"></a>
# ### Correlation matrix heatmap

# In[15]:


correlation_matrix = data.corr()

# Setting the correlation threshold
threshold = 0.30

# Creating a mask to display only correlations above the threshold
mask = np.triu(np.abs(correlation_matrix) >= threshold, k=1)

# Creating a heatmap with the "coolwarm" color palette
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, mask=~mask)
plt.title(f"Correlation Matrix Heatmap (Threshold = {threshold*100}%)")
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")

plt.show()


# ### The relevant feature names with their correlation percentages

# In[16]:


# Getting the correlations between features and the target variable, "SalePrice"
correlations_with_target = correlation_matrix['SalePrice']

# Filtering for features 
relevant_features = correlations_with_target[correlations_with_target >= threshold]

# Creating a list of tuples with feature names and their correlation percentages
correlation_tuples = [(feature, round(correlation * 100, 2)) for feature, correlation in relevant_features.items() if feature != 'SalePrice']

# Sorting the list by correlation percentages in descending order
correlation_tuples.sort(key=lambda x: x[1], reverse=True)

# the sorted relevant feature names with their correlation percentages
print("Features with correlations >= 30% with SalePrice (sorted):")
for feature, correlation_percentage in correlation_tuples:
    print(f"{feature}: {correlation_percentage}%")


# <a id="Outlier_detection"></a>
# ## Outlier detection(Normality test)

# In[17]:


def detect_outliers(data, threshold=1.5):
    outliers = None
    total_outliers = 0
    
    if pd.api.types.is_numeric_dtype(data):
        alpha = 0.05
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = stats.shapiro(data.dropna())

        if p > alpha:
            # Normal distribution, using Z-score method (Shapiro-Wilk test)
            z_scores = np.abs(stats.zscore(data))
            column_outliers = data[z_scores > threshold]
        else:
            # Non-normal distribution, use Tukey's method
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            column_outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        if column_outliers is not None:
            outliers = column_outliers
            total_outliers += len(column_outliers)
    
    return outliers, total_outliers


# In[18]:


def visualize_outliers(outliers):
    if outliers is not None:
        num_outliers = len(outliers.columns)
        num_rows = (num_outliers + 2) // 3  # Calculating the number of rows needed
        
        fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))
        axs = axs.flatten()  # Flatten the axis array to iterate over it
        
        for i, column in enumerate(outliers.columns):
            ax = axs[i]
            ax.boxplot(outliers[column].values, showfliers=False)
            ax.scatter(range(1, len(outliers)+1), outliers[column].values, color='red', marker='o', label='Outliers')
            ax.set_xlabel('Columns')
            ax.set_ylabel('Values')
            ax.set_title(f'Outliers - {column}')
            ax.legend()
        
        # Remove any unused subplots
        for j in range(num_outliers, len(axs)):
            fig.delaxes(axs[j])
        
        plt.tight_layout()
        plt.show()
    else:
        print('No outliers detected.')


#  <a id="Outlier_viz"></a>
# ### Visualizing outliers

# In[19]:


# Detect outliers for all numeric columns
all_outliers = pd.DataFrame()
total_outliers = 0
numeric_columns = data.select_dtypes(include=np.number).columns
for column in numeric_columns:
    column_data = data[column]
    column_outliers, column_total_outliers = detect_outliers(column_data)
    if column_outliers is not None:
        all_outliers[column] = column_outliers
        total_outliers += column_total_outliers

# Visualize outliers with a maximum of 3 graphs in a row
visualize_outliers(all_outliers)

# the number of total outliers in all columns
print("Number of total outliers:", total_outliers)


# <a id="Feature_Engineering"></a>
# # 3. Feature Engineering

# <a id="Feature_Engineering"></a>
# ### Checking if transformation is needed

# In[20]:


#Function to check whether transformation is needed or not.

def check_transformation_needed(data):
    for column in data.columns:
        original_skewness = data[column].skew()
        transformed_skewness = np.log1p(data[column]).skew()
        if transformed_skewness < original_skewness:
            print(f"Transformation recommended for column: {column}")
        else:
            print(f"No transformation needed for column: {column}")


# In[21]:


# Checking if transformation is needed
check_transformation_needed(data)


# <a id="Transformation"></a>
# ## Transformation

# In[22]:


# transformation

# numeric columns for transformation
columns_to_analyze = data.select_dtypes(include=np.number).columns

# Standardization
scaler = StandardScaler()
data[columns_to_analyze] = scaler.fit_transform(data[columns_to_analyze])

# Normalization
minmax_scaler = MinMaxScaler()
data[columns_to_analyze] = minmax_scaler.fit_transform(data[columns_to_analyze])


# ### Visualization before and after transformation

# In[23]:


# Histograms before standardization
data[columns_to_analyze].hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.title("Histograms Before Standardization")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Standardization
scaler = StandardScaler()
data[columns_to_analyze] = scaler.fit_transform(data[columns_to_analyze])

print("After standardization")

# Histograms after standardizati

data[columns_to_analyze].hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.title("Histograms After Standardization")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# <a id="Feature_importance"></a>
# ### Feature importance Analysis

# In[24]:


# Split the data into features and target variable
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you can proceed with training the Random Forest model
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Get feature importances
feature_importances = rf.feature_importances_

# Match feature importances with feature names
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

importance_df


# ### The correlation matrix of  important features

# In[25]:


important_features = ["OverallQual", "GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageArea", "MasVnrArea", "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd", "BsmtQual", "GarageYrBlt", "FullBath", "Fireplaces", "FireplaceQu", "GarageFinish", "GarageCars", "KitchenQual", "ExterQual"]

# Creating a subset of the dataset with important features and target variable
subset_data = data[important_features + ["SalePrice"]]

# the correlation matrix of important features
correlation_matrix = subset_data.corr()
correlation_matrix


# In[26]:


# A heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# ### The feature names and their correlations with "SalePrice"

# In[27]:


# Get correlations with "SalePrice" and sort them in descending order
target_correlations = correlation_matrix["SalePrice"].abs().sort_values(ascending=False)

# Print the feature names and their correlations with "SalePrice"
print("Correlations with SalePrice:")
for feature, correlation in target_correlations.items():
    print(f"{feature}: {correlation:.4f}")


# ### Visualizing important features

# In[28]:


# Get feature importances
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
feature_importances = gb_model.feature_importances_

# Match feature importances with feature names
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(14, 8))
plt.bar(importance_df['Feature'], importance_df['Importance'], color='b', alpha=0.7)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance for Gradient Boosting Model')
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.show()


# In[29]:


importance_df


# <a id="Feature_Selection"></a>
# ## Feature Selection and visualization

# In[30]:


# Subset the data with the selected important features and SalePrice
subset_data = data[important_features + ["SalePrice"]]

# Setting the style of the plots (optional but can make the plots look nicer)
sns.set(style="whitegrid")

num_plots = len(important_features)
num_cols = 3
num_rows = -(-num_plots // num_cols) 
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 20))

axes = axes.flatten() if num_rows > 1 else [axes]
for i, feature in enumerate(important_features):
    sns.scatterplot(x=feature, y="SalePrice", data=subset_data, ax=axes[i])

plt.tight_layout()
plt.show()


# ### Feature seleciton: 25 best features

# In[31]:


from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split

# feature selection
def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_regression, k='all')
    # learn the relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

# Set the random seed for reproducibility
random_state = 42

# Split the data into features and target variable
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)


# In[32]:


fs.scores_


# In[33]:


feature_names = fs.feature_names_in_
scores = fs.scores_

# Sort feature names and scores in descending order of scores
sorted_feature_names, sorted_scores = zip(*sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True))

plt.figure(figsize=(14, 8))  
plt.bar(sorted_feature_names, sorted_scores)
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.show()


# In[34]:


important_features = sorted_feature_names[:25]
print("Best 25 Features:")
print(important_features)


# In[35]:


important_features=['OverallQual', 'GrLivArea', 'ExterQual', 'BsmtQual', 'GarageCars', 'KitchenQual', 'GarageArea', '1stFlrSF', 'TotalBsmtSF', 'FullBath', 'GarageFinish', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'Fireplaces', 'MasVnrArea', 'FireplaceQu', 'HeatingQC', 'Foundation', 'BsmtFinSF1', 'GarageType', 'LotFrontage', 'OpenPorchSF', 'BsmtExposure']


# In[36]:


from sklearn.tree import DecisionTreeRegressor

# Initialize the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=random_state)

# Train the model
dt_model.fit(X_train, y_train)

# Get feature importances
feature_importances = dt_model.feature_importances_

# Match feature importances with feature names
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(14, 8))
plt.bar(importance_df['Feature'], importance_df['Importance'], color='b', alpha=0.7)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance for Decision Tree Model')
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.show()

# Get the top important features
important_features = importance_df['Feature'][:25].tolist()
print("Best 25 Features:")
print(important_features)


# <a id="Feature_rankings"></a>
# ## Feature rankings

# In[37]:


# Perform PCA to rank the features based on explained variance ratio
pca = PCA()
X_pca = pca.fit_transform(X)
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_df = pd.DataFrame({'Feature': X.columns, 'Explained Variance Ratio': explained_variance_ratio})

# Rank features using Random Forest
rf = RandomForestRegressor(random_state=random_state)
rf.fit(X, y)
feature_importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the feature importance dataframe in descending order
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# the feature rankings
print("Feature ranking using PCA - Explained Variance Ratio:")
print(explained_variance_df)
print("\nFeature ranking using Random Forest - Feature Importance:")
print(feature_importance_df)


# <a id="Model"></a>
# # 4.  Model Building and Evaluation

# In[38]:


# Perform PCA to rank the features based on explained variance ratio
pca = PCA()
X_pca = pca.fit_transform(X_train)
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_df = pd.DataFrame({'Feature': X_train.columns, 'Explained Variance Ratio': explained_variance_ratio})

# Rank features using Random Forest
rf = RandomForestRegressor(random_state=random_state)
rf.fit(X_train, y_train)
feature_importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Initialize different base models
linear_regression = LinearRegression()
decision_tree = DecisionTreeRegressor()
random_forest = RandomForestRegressor(random_state=random_state)
svr = SVR()
gradient_boosting = GradientBoostingRegressor()
ada_boost = AdaBoostRegressor()
ridge = Ridge()
lasso = Lasso()
elastic_net = ElasticNet()
knn = KNeighborsRegressor()
gaussian_process = GaussianProcessRegressor()

# Train each base model on the training set
linear_regression.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
svr.fit(X_train, y_train)
gradient_boosting.fit(X_train, y_train)
ada_boost.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
elastic_net.fit(X_train, y_train)
knn.fit(X_train, y_train)
gaussian_process.fit(X_train, y_train)

# Evaluate the performance of each model on the test set
linear_rmse = mean_squared_error(y_test, linear_regression.predict(X_test), squared=False)
dt_rmse = mean_squared_error(y_test, decision_tree.predict(X_test), squared=False)
rf_rmse = mean_squared_error(y_test, random_forest.predict(X_test), squared=False)
svr_rmse = mean_squared_error(y_test, svr.predict(X_test), squared=False)
gb_rmse = mean_squared_error(y_test, gradient_boosting.predict(X_test), squared=False)
ada_rmse = mean_squared_error(y_test, ada_boost.predict(X_test), squared=False)
ridge_rmse = mean_squared_error(y_test, ridge.predict(X_test), squared=False)
lasso_rmse = mean_squared_error(y_test, lasso.predict(X_test), squared=False)
enet_rmse = mean_squared_error(y_test, elastic_net.predict(X_test), squared=False)
knn_rmse = mean_squared_error(y_test, knn.predict(X_test), squared=False)
gp_rmse = mean_squared_error(y_test, gaussian_process.predict(X_test), squared=False)

mae_linear = mean_absolute_error(y_test, linear_regression.predict(X_test))
mae_dt = mean_absolute_error(y_test, decision_tree.predict(X_test))
mae_rf = mean_absolute_error(y_test, random_forest.predict(X_test))
mae_svr = mean_absolute_error(y_test, svr.predict(X_test))
mae_gb = mean_absolute_error(y_test, gradient_boosting.predict(X_test))
mae_ada = mean_absolute_error(y_test, ada_boost.predict(X_test))
mae_ridge = mean_absolute_error(y_test, ridge.predict(X_test))
mae_lasso = mean_absolute_error(y_test, lasso.predict(X_test))
mae_enet = mean_absolute_error(y_test, elastic_net.predict(X_test))
mae_knn = mean_absolute_error(y_test, knn.predict(X_test))
mae_gp = mean_absolute_error(y_test, gaussian_process.predict(X_test))

r2_linear = r2_score(y_test, linear_regression.predict(X_test))
r2_dt = r2_score(y_test, decision_tree.predict(X_test))
r2_rf = r2_score(y_test, random_forest.predict(X_test))
r2_svr = r2_score(y_test, svr.predict(X_test))
r2_gb = r2_score(y_test, gradient_boosting.predict(X_test))
r2_ada = r2_score(y_test, ada_boost.predict(X_test))
r2_ridge = r2_score(y_test, ridge.predict(X_test))
r2_lasso = r2_score(y_test, lasso.predict(X_test))
r2_enet = r2_score(y_test, elastic_net.predict(X_test))
r2_knn = r2_score(y_test, knn.predict(X_test))
r2_gp = r2_score(y_test, gaussian_process.predict(X_test))

# Print the evaluation metrics for each model
print("Linear Regression RMSE:", linear_rmse)
print("Decision Tree RMSE:", dt_rmse)
print("Random Forest RMSE:", rf_rmse)
print("SVR RMSE:", svr_rmse)
print("Gradient Boosting RMSE:", gb_rmse)
print("AdaBoost RMSE:", ada_rmse)
print("Ridge RMSE:", ridge_rmse)
print("Lasso RMSE:", lasso_rmse)
print("ElasticNet RMSE:", enet_rmse)
print("KNN RMSE:", knn_rmse)
print("Gaussian Process RMSE:", gp_rmse)
print()
print("Linear Regression MAE:", mae_linear)
print("Decision Tree MAE:", mae_dt)
print("Random Forest MAE:", mae_rf)
print("SVR MAE:", mae_svr)
print("Gradient Boosting MAE:", mae_gb)
print("AdaBoost MAE:", mae_ada)
print("Ridge MAE:", mae_ridge)
print("Lasso MAE:", mae_lasso)
print("ElasticNet MAE:", mae_enet)
print("KNN MAE:", mae_knn)
print("Gaussian Process MAE:", mae_gp)
print()
print("Linear Regression R-squared:", r2_linear)
print("Decision Tree R-squared:", r2_dt)
print("Random Forest R-squared:", r2_rf)
print("SVR R-squared:", r2_svr)
print("Gradient Boosting R-squared:", r2_gb)
print("AdaBoost R-squared:", r2_ada)
print("Ridge R-squared:", r2_ridge)
print("Lasso R-squared:", r2_lasso)
print("ElasticNet R-squared:", r2_enet)
print("KNN R-squared:", r2_knn)
print("Gaussian Process R-squared:", r2_gp)


# <a id="best_Model"></a>
# ## The best model and interpretation
# 
# Based on the evaluation results, the Gradient Boosting model emerges as the best performer among the models considered. Here's the interpretation:
# 
# 1. **Lowest RMSE and MAE**: The Gradient Boosting model achieved the lowest Root Mean Squared Error (RMSE) of approximately 0.325 and the lowest Mean Absolute Error (MAE) of approximately 0.208. These metrics indicate that, on average, the predictions are closer to the actual values compared to the other models.
# 
# 2. **Highest R-squared (R²)**: The Gradient Boosting model achieved the highest R-squared value of approximately 0.885. This means that the model explains about 88.5% of the variance in the target variable, indicating a very good fit.
# 
# 3. **Consistent Performance**: The Gradient Boosting model maintains good performance on both the training and test data. This suggests that it's not overfitting and can generalize well to unseen data.
# 
# In summary, the Gradient Boosting model is the preferred choice for this regression task, as it provides the most accurate predictions with the lowest error metrics and highest explained variance.

# <a id="compare"></a>
# ### Model Evaluation and Comparison

# In[39]:


models = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "SVR",
    "Gradient Boosting",
    "AdaBoost",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "KNN",
    "Gaussian Process"
]

# RMSE values
rmse_values = [
    0.3501114860784041,  # Linear Regression
    0.47199361339990664,  # Decision Tree
    0.36312351512291774,  # Random Forest
    0.3847572103333821,  # SVR
    0.32504872906989035,  # Gradient Boosting
    0.4156596730857431,  # AdaBoost
    0.3498049274109909,  # Ridge
    0.9595863194506697,  # Lasso
    0.7781306319078585,  # ElasticNet
    0.41462582751889077,  # KNN
    0.9579276541266011  # Gaussian Process
]

# MAE values
mae_values = [
    0.24656637808167833,  # Linear Regression
    0.30172041999306065,  # Decision Tree
    0.2380652809891943,  # Random Forest
    0.23511624743774195,  # SVR
    0.20806827561019622,  # Gradient Boosting
    0.30037212990387924,  # AdaBoost
    0.2463095386083,  # Ridge
    0.7341020833772043,  # Lasso
    0.5852813491848606,  # ElasticNet
    0.2823258896950631,  # KNN
    0.7295230622168355  # Gaussian Process
]

# R-squared values
r_squared_values = [
    0.866760212375211,  # Linear Regression
    0.7578449901462614,  # Decision Tree
    0.8566723561109904,  # Random Forest
    0.8390856599100561,  # SVR
    0.8851533957417516,  # Gradient Boosting
    0.8121993904788123,  # AdaBoost
    0.8669934405382761,  # Ridge
    -0.0008968201869419268,  # Lasso
    0.3418479501675922,  # ElasticNet
    0.8131324392338433,  # KNN
    0.002560332151043232  # Gaussian Process
]

# Plotting RMSE
plt.figure(figsize=(10, 6))
plt.bar(models, rmse_values, color='b', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Root Mean Squared Error (RMSE) for Regression Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting MAE
plt.figure(figsize=(10, 6))
plt.bar(models, mae_values, color='g', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('MAE')
plt.title('Mean Absolute Error (MAE) for Regression Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting R-squared
plt.figure(figsize=(10, 6))
plt.bar(models, r_squared_values, color='r', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('R-squared')
plt.title('R-squared for Regression Models')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# <a id="Tuning"></a>
# ## Hyperparameter Tuning and Model Selection
# From what we've seen before, it seems like Gradient Boosting and RandomForest are doing really well. Now, let's try to make them even better by adjusting some settings.

# In[40]:


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Parameter grids for both models
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search for Gradient Boosting
grid_search_gb = GridSearchCV(estimator=gradient_boosting, param_grid=gb_param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_gb.fit(X_train, y_train)

best_params_gb = grid_search_gb.best_params_
best_model_gb = grid_search_gb.best_estimator_
best_score_gb = grid_search_gb.best_score_

# Grid Search for Random Forest
grid_search_rf = GridSearchCV(estimator=random_forest, param_grid=rf_param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)

best_params_rf = grid_search_rf.best_params_
best_model_rf = grid_search_rf.best_estimator_
best_score_rf = grid_search_rf.best_score_

# Compare results
if best_score_gb < best_score_rf:
    best_model = best_model_gb
    best_params = best_params_gb
    best_score = best_score_gb
else:
    best_model = best_model_rf
    best_params = best_params_rf
    best_score = best_score_rf

print(f"Best Model: {type(best_model).__name__}")
print(f"Best Parameters: {best_params}")


# <a id="using_test_data"></a>
# ## Model Prediction using test data

# In[101]:


# Loading test data

test_data = pd.read_csv('test.csv')

def Label_Encoder(df):
    object_cols = df.select_dtypes(include='object').columns
    for col in object_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

test_data = Label_Encoder(test_data)


from DataFrame_Checker import DataFrameChecker

# An instance of DataFrameChecker
checker = DataFrameChecker(test_data)

# Dropping rows with missing values
test_data = test_data.fillna(test_data.mode().iloc[0])


# Initialize the model
best_model = RandomForestRegressor()


# Define important features
important_features = ['OverallQual', 'GrLivArea', 'ExterQual', 'BsmtQual', 'GarageCars', 'KitchenQual', 'GarageArea', 
                      '1stFlrSF', 'TotalBsmtSF', 'FullBath', 'GarageFinish', 'TotRmsAbvGrd', 'YearBuilt', 
                      'YearRemodAdd', 'GarageYrBlt', 'Fireplaces', 'MasVnrArea', 'FireplaceQu', 'HeatingQC', 
                      'Foundation', 'BsmtFinSF1', 'GarageType', 'LotFrontage', 'OpenPorchSF', 'BsmtExposure']

train_data=pd.read_csv('train.csv')
data = Label_Encoder(train_data)
# Prepare training and testing data


data=data.fillna(data.mean())

X_train = data[important_features]
X_test = test_data[important_features]
y_train = data["SalePrice"]

# Train the model
best_model.fit(X_train, y_train)

# Make predictions
predictions = best_model.predict(X_test)

# Create a DataFrame for the results
results = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions})
print(results)

# Save the results as submission.csv
results.to_csv('submission1.csv', index=False)


# In[107]:


# Initialize the model
best_model = GradientBoostingRegressor()

# Load and preprocess training data
train_data = pd.read_csv('train.csv')
data = Label_Encoder(train_data)

# Fill missing values with means
data = data.fillna(data.mean())

# Separate features and target variable
X_train = data.drop("SalePrice", axis=1)
y_train = data["SalePrice"]

# Train the model
best_model.fit(X_train, y_train)

# Make predictions using all features from test_data
predictions = best_model.predict(test_data)

# Create a DataFrame for the results
results = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions})
print(results)

# Save the results as submission.csv
results.to_csv('submission_gradient_boosting2.csv', index=False)


# In[111]:


# Initialize the model
best_model = RandomForestRegressor(max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300)

# Load and preprocess training data
train_data = pd.read_csv('train.csv')
data = Label_Encoder(train_data)

# Fill missing values with means
data = data.fillna(data.mean())

# Separate features and target variable
X_train = data.drop("SalePrice", axis=1)
y_train = data["SalePrice"]

# Train the model
best_model.fit(X_train, y_train)

# Make predictions using all features from test_data
predictions = best_model.predict(test_data)

# Create a DataFrame for the results
results = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions})
print(results)

# Save the results as submission.csv
results.to_csv('submission_RandomForestRegressor3.csv', index=False)


# In[112]:


# Initialize the model
best_model = GradientBoostingRegressor(max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=300)

# Load and preprocess training data
train_data = pd.read_csv('train.csv')
data = Label_Encoder(train_data)

# Fill missing values with means
data = data.fillna(data.mean())

# Separate features and target variable
X_train = data.drop("SalePrice", axis=1)
y_train = data["SalePrice"]

# Train the model
best_model.fit(X_train, y_train)

# Make predictions using all features from test_data
predictions = best_model.predict(test_data)

# Create a DataFrame for the results
results = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions})
print(results)

# Save the results as submission.csv
results.to_csv('submission_GradientBoostingRegressor3.csv', index=False)


# <a id="Pytorch"></a>
# 
# # 5. Deep Learning (Pytorch Lightening)

# In[43]:


# Defining LightningModule
class HousePriceLightningModel(pl.LightningModule):
    def __init__(self, input_dim):
        super(HousePriceLightningModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)  # Log the validation loss
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Loading Data
data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

# Checking for Missing Values
missing_values = data.isnull().sum()

# Data Preprocessing

# Defining Label Encoder Function
def label_encoder(df):
    object_cols = df.select_dtypes(include='object').columns
    for col in object_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

# Applying Label Encoder
data = label_encoder(data)

# Handling Missing Values
data = data.fillna(data.mean())

# Defining Features and Target
X = data.drop(['Id', 'SalePrice'], axis=1)
y = data['SalePrice']

# Converting to PyTorch Tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)


# Defining PyTorch Dataset class

from torch.utils.data import Dataset

class HousePriceDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.target[idx]
        return x, y
    
    

# Initializing LightningModule

input_dim = X.shape[1]
lightning_model = HousePriceLightningModel(input_dim)



# Creating DataLoaders

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = HousePriceDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = HousePriceDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32)



# Initialize a Trainer
trainer = pl.Trainer(max_epochs=10)

# Train the Model
trainer.fit(lightning_model, train_loader, val_loader)



# Initialize empty lists to store training and validation losses
train_losses = []
val_losses = []
optimizer = torch.optim.Adam(lightning_model.parameters(), lr=0.001)

for epoch in range(100):
    train_loss_sum = 0.0
    val_loss_sum = 0.0
    
    # Training Step
    for batch in train_loader:
        x, y = batch
        y_hat = lightning_model(x)
        loss = nn.MSELoss()(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()
    
    # Validation Step
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            y_hat = lightning_model(x)
            loss = nn.MSELoss()(y_hat, y)
            val_loss_sum += loss.item()
    
    # average loss for the epoch
    train_loss_avg = train_loss_sum / len(train_loader)
    val_loss_avg = val_loss_sum / len(val_loader)   
    

    # Store the losses for plotting
    train_losses.append(train_loss_avg)
    val_losses.append(val_loss_avg)   
    
    
# Saving the trained model
torch.save(lightning_model.state_dict(), "house_price_model.pt")


# <a id="Losses"></a>
# ## Visualizing the Losses(Train VS Validation Loss)

# In[44]:


# Visualizing the Losses
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', color='blue')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# <a id="Final_Prediction"></a>
# 
# ## End Result

# In[45]:


# Defining prediction function

def predict():
    # Load the trained model
    model = HousePriceLightningModel(input_dim)
    model.load_state_dict(torch.load("house_price_model.pt"))
    
    # Loading test data and preprocess
    test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
    test_data = label_encoder(test_data)
    test_data = test_data.fillna(test_data.mean())
    X_test = test_data.drop('Id', axis=1)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    
    # Ensuring the Model is in Evaluation Mode
    model.eval()
    
    # Making Predictions
    with torch.no_grad():
        predictions = model(X_test)
    
    return predictions.numpy()

print("Saving the trained model...")
torch.save(lightning_model.state_dict(), "house_price_model.pt")

print("Making Predictions...")
predictions = predict()

predictions = predictions.reshape(-1)

# Create a DataFrame for the results
results = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': predictions})
results


# 
# <div style="text-align: center;">
#     <img src="https://www.hearthaustralia.com.au/wp-content/uploads/2020/09/Hearth-Housing-A_Obq.png" alt="Image" style="width: 50%; height: auto;">
# </div>
# 
# 
# 
# <div style = "text-align: center"><font size = 5 color = "#00008B" face = "verdana"><b>The total number of homeless people in the world: 122292435. </b></font></div><br/> 
# <div style = "text-align: center"><font size = 6 color = "#B22222" face = "verdana"><b>We envision a world without homeless people.</b></font></div> <br/> 
# 
