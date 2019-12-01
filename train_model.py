import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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
classifier = LinearDiscriminantAnalysis()

# Training model
classifier.fit(X, y)

# Save the model
filename = 'trained_LDA'
joblib.dump(classifier, filename)

