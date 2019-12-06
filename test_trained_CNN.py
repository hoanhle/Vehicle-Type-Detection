import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# CHANGE THESE
all_dir = 'all_images/'
test_dir = "testset/"
model_path = 'my_model.h5'

# Create an index of class names

class_names = sorted(os.listdir(all_dir))

trained_model = load_model(model_path)

with open("submissionCNN.csv", "w") as fp:
    fp.write("Id,Category\n")

    # Image index
    i = 0
    # 1. load image and resize
    for file in sorted(os.listdir(test_dir)):
        if file.endswith(".jpg"):
            # Load the image
            img = plt.imread(test_dir + file)
            # Resize it to the net input size:
            img = cv2.resize(img, (224, 224))
            img = img[np.newaxis, ...]

            # Convert the data to float:
            img = img.astype(np.float32)

            # Predict class by picking the highest probability index
            # then add 1 (due to indexing behavior)
            class_index = np.argmax(trained_model.predict(img)[0]) + 1

            # Convert class id to name
            label = class_names[class_index]

            fp.write("%d,%s\n" % (i, label))

            print(i)
            i += 1
