'''
Data Analytics III
1. Implement Simple Naïve Bayes classification algorithm using Python/R on iris.csv dataset.
2. Compute Confusion matrix to find TP, FP, TN, FN, Accuracy, Error rate, Precision, Recall
on the given dataset
'''
import pandas as pd
import matplotlib.pyplot as plt

data= pd.read_csv('iris.csv')
print(data.isnull().sum())

X = data.drop(['Species', 'Id'], axis=1)
y = data.drop(['Id',  'SepalLengthCm',  'SepalWidthCm',  'PetalLengthCm', 'PetalWidthCm'], axis=1)
print(X)
print(y)
print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
model.score(X_test,y_test)

conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate TP, FP, TN, FN
TP = conf_matrix[1, 1]
FP = conf_matrix[0, 1]
TN = conf_matrix[0, 0]
FN = conf_matrix[1, 0]

# Compute Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Compute Error Rate
error_rate = 1 - accuracy

# Compute Precision
precision = precision_score(y_test, y_pred, average='macro')

# Compute Recall
recall = recall_score(y_test, y_pred, average='macro')

print("Confusion Matrix:")
print(conf_matrix)
print("True Positives:", TP)
print("False Positives:", FP)
print("True Negatives:", TN)
print("False Negatives:", FN)
print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)



###########################

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
print("Confusion matrix:")
print(cm)

def get_confusion_matrix_values(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return(cm[0][0], cm[0][1], cm[1][0], cm[1][1])

TP, FP, FN, TN = get_confusion_matrix_values(y_test, y_pred)
print("TP: ", TP)
print("FP: ", FP)
print("FN: ", FN)
print("TN: ", TN)

print("The Accuracy is ", (TP+TN)/(TP+TN+FP+FN))
print("The precision is ", TP/(TP+FP))
print("The recall is ", TP/(TP+FN))

# f1 score being at lower side 





'''
# extra 
df['col1'].value_counts()
total_yes = manually
total_no = manually


df.crosstab(data[col1], data[predict_col])

pon = mat.loc['row', 'col']/total_no

# Using .loc[]
count_setosa_as_setosa = conf_matrix.loc['setosa', 'setosa']
'''