import scipy.io as sio
import numpy as np
from sklearn.svm import SVC
#import cv2
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
SVC_rbf = SVC(kernel='rbf')
 
# Training model
SVC_rbf.fit(X, y)
 
 
# Save the model
filename = 'trained_SVC_rbf'
joblib.dump(SVC_rbf, filename)