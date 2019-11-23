import scipy.io as sio
import numpy as np
from sklearn.svm import SVC
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.externals import joblib

# Create an index of class names

class_names = sorted(os.listdir("train\\train"))

# Prepare a pretrained CNN for feature extraction

base_model = tf.keras.applications.mobilenet.MobileNet(
    input_shape=(224, 224, 3),
    include_top=False)

#
in_tensor = base_model.inputs[0]
out_tensor = base_model.outputs[0]

out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)

# Define the full model by the endpoints
model = tf.keras.models.Model(inputs=[in_tensor], outputs=[out_tensor])

# Compile the model for execution. Losses and optimizers can be
# anything here, since we don't train the model
model.compile(loss="categorical_crossentropy", optimizer='sgd')

SVC_rbf = joblib.load('trained_SVC_rbf')

with open("submission_SVC_rbf.csv", "w") as fp:
    fp.write("Id,Category\n")

    # Image index
    i = 0
    # 1. load image and resize
    for file in os.listdir("test\\testset"):
        if file.endswith(".jpg"):
            # Load the image
            img = plt.imread("test\\testset\\" + file)

            # Resize it to the net input size:
            img = cv2.resize(img, (224, 224))

            # Convert the data to float, and remove mean:
            img = img.astype(np.float32)
            img -= 128

            # 2. vectorize using the net

            x = model.predict(img[np.newaxis, ...])


            # 3. predict class using the sklearn model
            class_index = SVC_rbf.predict(x)[0]

            # 4. convert class id to name (label = class_names[class_index])
            label = class_names[class_index]

            fp.write("%d,%s\n" % (i, label))

            print(i)
            i += 1