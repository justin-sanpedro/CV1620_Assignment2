import cv2
import glob
import numpy as np
import joblib

path = "src\\Assignment2\\Assignment2\\Q1\\"
#process sample images
for i, address in enumerate(glob.glob(path + "sample/*")):
    img = cv2.imread(address)
    img = cv2.resize(img, (500, 500))
    image = cv2.resize(img, (32,32))
    image = image/255
    image = image.flatten()

    #knn
    model = joblib.load("cat_dog_classification_knn.z")
    prediction = model.predict([image])[0]
    cv2.putText(img, f"KNN: {prediction}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

    #log reg
    model = joblib.load("cat_dog_classification_log_reg.z")
    prediction = model.predict([image])[0]
    cv2.putText(img, f"Logisic Regression: {prediction}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)

    #sgd
    model = joblib.load("cat_dog_classification_sgd.z")
    prediction = model.predict([image])[0]
    cv2.putText(img, f"SGD: {prediction}", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

    cv2.imshow('frame', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

"""
The model isn't very accurate. Although KNN scored high accuracy during the test data with 70%,
it only is able to identify 1 out of 6 of the sample images taken from the internet.

On the other hand, Logistic Regression and Stochastic Gradient Descent is able to correctly identify
4 out of 6 from the sample data even if they only scored 40% in the test data.

Overall, the three models are very inconsistent and so they are not very accurate with their 
predictions.
"""
