import scipy.io as sio
import numpy as np
from sklearn.svm import SVC
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.externals import joblib


# Create an index of class names

class_names = sorted(os.listdir(
    r"D:\Downloads\DOCUMENTS\STUDIES\ml\pattern_recognition_ml\Code\project\train\train"))

# Prepare a pretrained CNN for feature extraction

base_model = tf.keras.applications.mobilenet.MobileNet(
    input_shape=(224, 224, 3),
    include_top=False)


# Get the network structure
print(base_model.summary())

#
in_tensor = base_model.inputs[0]
out_tensor = base_model.outputs[0]

out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)

# Define the full model by the endpoints
model = tf.keras.models.Model(inputs=[in_tensor], outputs=[out_tensor])

# Compile the model for execution. Losses and optimizers can be
# anything here, since we don't train the model
model.compile(loss="categorical_crossentropy", optimizer='sgd')

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
filename = 'trained_SVC'
joblib.dump(SVC_rbf, filename)

