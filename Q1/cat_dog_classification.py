import numpy as np
import glob
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import joblib

# DATA
data_list_train = []
label_list_train = []
data_list_test = []
label_list_test = []

path = "src\\Assignment2\\Assignment2\\Q1\\"
#process images for training
for i, address in enumerate(glob.glob(path + "train/*/*")):
    image = cv2.imread(address)
    image = cv2.resize(image, (32,32))
    image = image/255
    image = image.flatten()

    data_list_train.append(image)

    label_list_train.append(address.split('\\')[-2])

    if i%200 == 0:
        print(f'[INFO] {i} images read for Training!')

#process images for testing
for i, address in enumerate(glob.glob(path + "test/*/*")):
    image = cv2.imread(address)
    image = cv2.resize(image, (32,32))
    image = image/255
    image = image.flatten()

    data_list_test.append(image)

    label_list_test.append(address.split('\\')[-2])

X_train = np.array(data_list_train)
y_train = np.array(label_list_train)
X_test = np.array(data_list_test)
y_test = np.array(label_list_test)

# MODEL

#KNN
random_state = 42
k = 19
model_knn = KNeighborsClassifier(n_neighbors=k)
model_knn.fit(X_train, y_train)

#Logistic Regression
model_log_reg = LogisticRegression(max_iter=1000, random_state=random_state)
model_log_reg.fit(X_train, y_train)

#Stochastic Gradient Descent
model_sgd = SGDClassifier(random_state=random_state)
model_sgd.fit(X_train, y_train)


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

joblib.dump(model_knn, "cat_dog_classification_knn.z")
joblib.dump(model_log_reg, "cat_dog_classification_log_reg.z")
joblib.dump(model_sgd, "cat_dog_classification_sgd.z")