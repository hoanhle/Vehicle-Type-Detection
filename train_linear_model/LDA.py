import scipy.io as sio
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
LDA_clf = LinearDiscriminantAnalysis()
 
# Training model
LDA_clf.fit(X, y)
 
 
# Save the model
filename = 'trained_LDA'
joblib.dump(LDA_clf, filename)