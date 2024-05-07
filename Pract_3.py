# Import necessary libraries
import pandas as pd
import seaborn as sns

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')
print(titanic.head())                   # for first 5 rows
print(titanic.tail())                   # for last 5 rows
print(titanic.info())                   # info of dataset
print(titanic.describe())               # describe data
print(titanic.shape )                   # dimension of data, no. of rows and cols
print(titanic.dtypes)                   # types of each cols

print(titanic.isnull().sum())           # no of nulls in each cols
print(titanic.notnull().sum())          # no of non nulls in each cols


'''
titanic.dropna(axis = 0, subset = [titanic["col1"], titanic["col2"],...], inplace = True)                 # use this if null values are less
             
x = titanic["col1"].mean()     # if col dtype is int, float
titanic["col1"].fillna(value = x, axis = 0, inplace = True)           # use if null values are more

# if col dtype is not int, float
x = titanic["col1"].value_counts().idmax()   
titanic["col1"].fillna(value = x, axis = 0, inplace = True)   

print(titanic.isnull().sum())           # no of nulls in each cols
titanic.duplicated()             # print duplicates
titanic.drop_duplicates()
'''

# 1. Provide summary statistics for a dataset with numeric variables grouped by one of the qualitative variable
# Here, we group by 'Sex' and provide summary statistics for 'Age'
data = titanic.groupby('sex')['age']

# Summary statistics
print("Summary statistics for Age grouped")
print("Mean\n", data.mean())
print("Median\n", data.median())
print("Minimum\n", data.min())
print("Maximum\n", data.max())
print("Standard Deviation\n", data.std())

numeric = [data.mean(), data.median(), data.min(), data.max(), data.std()]
print("List of numeric values for each response to the categorical variable\n", numeric)

iris = sns.load_dataset('iris')

iris_setosa = iris[iris['species'] == 'setosa'].describe()
iris_versicolor = iris[iris['species'] == 'versicolor'].describe()
iris_virginica = iris[iris['species'] == 'virginica'].describe()

print("Basic statistical details of the species of ‘Iris-setosa’\n", iris_setosa)
print("Basic statistical details of the species of ‘Iris-versicolor’\n", iris_versicolor)
print("Basic statistical details of the species of ‘Iris-virginica’\n", iris_virginica)