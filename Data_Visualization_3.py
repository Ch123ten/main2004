
# Title: Data Visualization-III

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read the data from the file
df = sns.load_dataset('titanic')

# 1. List down the features and their types (e.g., numeric, nominal) available in the dataset.
# Feature names and types
print("Features:")
for feature in df.columns:
    if df[feature].dtype == 'int64' or df[feature].dtype == 'float64':
        print(f"Type of {feature}: Numeric")
    else:
        print(f"Type of {feature}: Nominal")
    if df[feature].dtype == 'object':
        print(f"  Unique values: {df[feature].unique()}")

        
# 2. Create a histogram for each feature in the dataset to illustrate the feature distributions.

for feature in df.columns:
    if df[feature].dtype == 'int64' or df[feature].dtype == 'float64':
        plt.hist(df[feature], bins=20)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


# 3. Create a box plot for each feature in the dataset.

for feature in df.columns:
    if df[feature].dtype == 'int64' or df[feature].dtype == 'float64':
        plt.boxplot(df[feature])
        plt.title(f"Box plot of {feature}")
        plt.show()


# 4. Compare distributions and identify outliers.

sns.pairplot(df, hue='class')
plt.show()

# Find Outliers numerically
for feature in df.columns:
    if df[feature].dtype == 'int64' or df[feature].dtype == 'float64':
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        print(f"Outliers for {feature}: {df[(df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR))]}")




