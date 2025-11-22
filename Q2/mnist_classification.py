import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import joblib

"""
This script creates the trained models.

The tests are conducted in mnist_classification_inference.py.
The trained models are required before running the tests.
"""

# DATA
data_list_train = []
label_list_train = []
data_list_test = []
label_list_test = []

num_training = 60000

path = "src\\Assignment2\\Assignment2\\Q2\\"
#process images for training
df_train = pd.read_csv(path+'mnist_train.csv', sep=',', header=None)
df_train = df_train.iloc[:num_training, :]

#get first column as label
y_train = df_train.pop(df_train.columns[0])
y_train = np.array(y_train)
#rest of the column as flattened image
X_train = df_train
X_train = np.array(X_train)
X_train = X_train/255.0 #normalize values

# MODEL

#KNN
random_state = 42
k = 9
model_knn = KNeighborsClassifier(n_neighbors=k)
model_knn.fit(X_train, y_train)

#Logistic Regression
model_log_reg = LogisticRegression(max_iter=5000, random_state=random_state)
model_log_reg.fit(X_train, y_train)

#Stochastic Gradient Descent
model_sgd = SGDClassifier(random_state=random_state)
model_sgd.fit(X_train, y_train)


# EVALUATE

joblib.dump(model_knn, "mnist_classification_knn.z")
joblib.dump(model_log_reg, "mnist_classification_log_reg.z")
joblib.dump(model_sgd, "mnist_classification_sgd.z")