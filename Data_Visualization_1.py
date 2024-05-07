import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
titanic = sns.load_dataset('titanic')
print(titanic.shape)

print(titanic.columns)

# Visualize the distribution of 'age' column
sns.histplot(data=titanic, x="age", kde=True)
plt.show()

# Display the distribution of 'sex' and 'class'
sns.countplot(x='sex', data=titanic)
plt.show()

sns.countplot(x='class', data=titanic)
plt.show()

# Explore the relationship between 'fare' and 'class'
sns.boxplot(x='class', y='fare', data=titanic)
plt.show()

# Explore the relationship between 'age' and 'class'
sns.boxplot(x='class', y='age', data=titanic)
plt.show()

# Visualize the survival rate based on 'class'
sns.barplot(x="class", y="survived", data=titanic)
plt.show()

	
# Explore the survival rate by 'sex' and 'class'
sns.catplot(x='class', y='survived', hue='sex', kind='bar', data=titanic)
plt.show()

# visualize the survival rate based on sex
sns.barplot(x="sex",y="survived",data=titanic)
plt.show()

# 2. Write a code to check how the price of the ticket (column name: 'fare') for each passenger is
# distributed by plotting a histogram

sns.histplot(data=titanic, x='fare', kde=True, bins=30)
plt.title('Distribution of Fares')  # Set the title of the plot
plt.show()

