import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Read the data from the CSV file
data = pd.read_csv('data.csv')

print(data.head())
# 1. Scan all variables for missing values and inconsistencies. If there are missing values and/or 
# inconsistencies, use any of the suitable techniques to deal with them.

# Check for missing values
print(data.isnull().sum())
# Check for inconsistencies
print(data.describe())

# fill missing values with the mean of the column on CGPA1 and CGPA2
data['CGPA1'] = data['CGPA1'].fillna(data['CGPA1'].mean())
data['CGPA2'] = data['CGPA2'].fillna(data['CGPA2'].mean())

print(data.isnull().sum())

# 2. Scan all numeric variables for outliers. If there are outliers, use any of the suitable techniques 
# to deal with them.

# Check for outliers
sns.boxplot(data['age'])
plt.show()

# look for outliers in the age column
Q1 = data['age'].quantile(0.25)
Q3 = data['age'].quantile(0.75)

IQR = Q3 - Q1

print("Q1: ", Q1)
print("Q3: ", Q3)
print("IQR: ", IQR)

# print the number of outliers
outliers = data[(data['age'] < (Q1 - 1.5 * IQR)) | (data['age'] > (Q3 + 1.5 * IQR))]
print(outliers)

data['age'] = data['age'].mask(data['age'] > Q3 + 1.5 * IQR, data['age'].mode()[0])
data['age'] = data['age'].mask(data['age'] < Q1 - 1.5 * IQR, data['age'].mode()[0])

print(data['age'])

# 3. Apply data transformations on at least one of the variables. The purpose of this transformation 
# should be one of the following reasons: to change the scale for better understanding of the 
# variable, to convert a non-linear relation into a linear one, or to decrease the skewness and 
# convert the distribution into a normal distribution.
# Reason and document your approach properly.

# The age column has a centered data. We can apply a log transformation to the age column to 
# convert the distribution into a normal distribution.

# log transformation
data['age'] = data['age'].apply(lambda x: np.log(x) if x > 0 else 0)

# display the transformed data
print(data['age'])

# show age distribution after transformation in boxplot
sns.boxplot(data['age'])
plt.show()