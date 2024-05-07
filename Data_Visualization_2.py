
# 1. Use the inbuilt dataset 'titanic' as used in the above problem. Plot a box plot for distribution of 
# age  with  respect to each gender along with  the information about whether they survived or  not. 
# (Column names : 'sex' and 'age') 

# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the Titanic dataset
titanic = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(titanic.head(10))

titanic.isnull().sum()

m_val   = titanic["age"].astype("float").mean(axis = 0)
print("mean of stroke  is: ",m_val)
titanic["age"].replace(np.nan,m_val,inplace = True)

var1= titanic['deck'].value_counts().idxmax()

titanic['deck'].fillna(value=var1, inplace=True)

var2= titanic['embarked'].value_counts().idxmax()

titanic['embarked'].fillna(value=var2, inplace=True)

var3= titanic['embark_town'].value_counts().idxmax()

titanic['embark_town'].fillna(value=var3, inplace=True)
print(titanic.isnull().sum())

# Count the number of survivors and non-survivors for each gender
male_survivors = titanic[(titanic['sex'] == 'male') & (titanic['survived'] == 1)].shape[0]
male_non_survivors = titanic[(titanic['sex'] == 'male') & (titanic['survived'] == 0)].shape[0]
female_survivors = titanic[(titanic['sex'] == 'female') & (titanic['survived'] == 1)].shape[0]
female_non_survivors = titanic[(titanic['sex'] == 'female') & (titanic['survived'] == 0)].shape[0]

print(f"Male survivors: {male_survivors}")
print(f"Male non-survivors: {male_non_survivors}")
print(f"Female survivors: {female_survivors}")
print(f"Female non-survivors: {female_non_survivors}")

# Plot a box plot of the 'age' column with respect to 'sex' and 'survived'
plt.figure(figsize=(10,6))  # Set the figure size
sns.boxplot(x='sex', y='age', hue='survived', data=titanic)
plt.title('Box Plot of Age Distribution by Sex and Survival')  # Set the title of the plot
plt.show()
'''
	
2. Observations
1. The median age of survivors and non-survivors is slightly different for both
2. More females are survivors than males
3. There are outliers in male non survivors


'''