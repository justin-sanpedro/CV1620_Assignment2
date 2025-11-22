import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# DATA

#process images for testing
path = "src\\Assignment2\\Assignment2\\Q2\\"
df_test = pd.read_csv(path+'mnist_test.csv', sep=',', header=None)

#get first column as label
y_test = df_test.pop(df_test.columns[0])
y_test = np.array(y_test)
#rest of the column as flattened image
X_test = df_test
X_test = np.array(X_test)
X_test = X_test/255.0 #normalize values

# MODEL
#load trained model using KNN
model_knn = joblib.load("mnist_classification_knn.z")
#load trained model using logistic regression
model_log_reg = joblib.load("mnist_classification_log_reg.z")
#load trained model using stochastic gradient descent
model_sgd = joblib.load("mnist_classification_sgd.z")


# EVALUATE
#get accuracy for KNN model
predictions_knn = model_knn.predict(X_test)
print(f"K Nearest Neighbors Accuracy: {accuracy_score(y_test, predictions_knn)*100}%")

#get accuracy for logistic regression model
predictions_log_reg = model_log_reg.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, predictions_log_reg)*100}%")

#get accuracy for stochastic gradient descent model
predictions_sgd = model_sgd.predict(X_test)
print(f"Stochastic Gradient Descent Accuracy: {accuracy_score(y_test, predictions_sgd)*100}%")

"""
NOTE: Make sure to have the three mnist trained models in the root directory 
(run mnist_classification.py to create the training models)

The model yielding the highest accuracy score is the one that uses K Nearest Neighbor 
with an accuracy of 96.59% from the 10000 test cases provided.
The second one with the highest accuracy score is the Logistic Regression, but only by a around 1%
above the one that used Stochastic Gradient Descent.

All models have an accuracy score of above 90%.
"""
