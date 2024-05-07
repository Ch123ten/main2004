import pandas as  p
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

data = p.read_csv('BostonHousing.csv')
print(data.head())
print(data.tail())
print("The shape of the data is: ")
print(data.shape)
print(data.isnull().sum())
data.dropna(subset= ['rm'], inplace=True)
print(data.isnull().sum())



X = data.iloc[:,0:13]
y = data.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


from sklearn.metrics import mean_squared_error, r2_score

# Evaluate the model accuracy
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Mean Squared Error:", mse)
print("R-squared:", r2)

print(model.score(X_test,y_test))
 


# mse is sum of ( difference in actual and predicted y values ) / total number of points