
# 1. Import all the required Python Libraries.
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# 2. Locate an open source data from the web (e.g., https://www.kaggle.com). Provide a clear
# description of the data and its source (i.e., URL of the web site).
# Download dataset from kaggle.com and import in to file where it needed 


# 3. Load the Dataset into pandas dataframe.
df = pd.read_csv("file-name.csv")


# 4. Data Preprocessing: check for missing values in the data using pandas isnull(), describe()
# function to get some initial statistics. Provide variable descriptions. Types of variables etc.
# Check the dimensions of the data frame.
print(df.head())                   # for first 5 rows
print(df.tail())                   # for last 5 rows
print(df.info())                   # info of dataset
print(df.describe())               # describe data
print(df.shape )                   # dimension of data, no. of rows and cols
print(df.dtypes)                   # types of each cols

print(df.isnull().sum())           # no of nulls in each cols
print(df.notnull().sum())          # no of non nulls in each cols

'''

df.dropna(axis = 0, subset = [df["col1"], df["col2"],...], inplace = True)                 # use this if null values are less
             
x = df["col1"].mean()     # if col dtype is int, float
df["col1"].fillna(value = x, axis = 0, inplace = True)           # use if null values are more

# if col dtype is not int, float
x = df["col1"].value_counts().idmax()   
df["col1"].fillna(value = x, axis = 0, inplace = True)   

print(df.isnull().sum())           # no of nulls in each cols
df.duplicated()             # print duplicates
df.drop_duplicates()

# Reset index
df.reset_index(drop=True, inplace=True)

'''



# 5. Data Formatting and Data Normalization: Summarize the types of variables by checking
# the data types (i.e., character, numeric, integer, factor, and logical) of the variables in the
# data set. If variables are not in the correct data type, apply proper type conversions.
print(df.dtypes)

df["col"] = df["col"].astype(dtype=int, errors = 'coerce')    # dtype =  int, float, str, 'category','bool','object','datetime64'



# 6. Turn categorical variables into quantitative variables in Python.
df["col"] = pd.Categorical(df["col"])
df["col"] = df["col"].cat.codes
print(df["col"])


'''
sex
male            0
female          1
trans           2

'''