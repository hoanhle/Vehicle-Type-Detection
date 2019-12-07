import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from combine_model import ensemble_predictions

# CHANGE THESE
all_dir = 'train/train'
test_dir = "testset/"

# Create an index of class names

class_names = sorted(os.listdir(all_dir))


"""
TODO: Change model paths
"""
model1 = load_model("my_model.h5")
model2 = load('./InceptionV3-model')

models = [model1, model2]


with open("submissionCNN.csv", "w") as fp:
    fp.write("Id,Category\n")

    # Image index
    i = 0
    # 1. load image and resize
    for file in os.listdir(test_dir):
        print(file)
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
            class_index = ensemble_predictions(models, img)

            # Convert class id to name
            label = class_names[class_index]

            fp.write("%d,%s\n" % (i, label))

            print(i)
            i += 1
