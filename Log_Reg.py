import scipy.io as sio
import numpy as np
from sklearn.linear_model import LogisticRegression
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.externals import joblib


# Loading mat file from the folder
mat_contents = sio.loadmat("features.mat")
X = mat_contents['X']
y = mat_contents['y'].ravel()

print(X.shape)
print(y.shape)

# Define classifier
logistic_regression = LogisticRegression()

# Training model
logistic_regression.fit(X, y)


# Save the model
filename = 'trained_LOG_REG'
joblib.dump(logistic_regression, filename)

