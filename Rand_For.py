import scipy.io as sio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
rand_forest = RandomForestClassifier(n_estimators=100)

# Training model
rand_forest.fit(X, y)


# Save the model
filename = 'trained_RAND_FOR'
joblib.dump(rand_forest, filename)

