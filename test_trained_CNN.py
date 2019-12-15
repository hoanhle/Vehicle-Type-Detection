import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import load_model


# CHANGE THESE
all_dir = 'train/train'
test_dir = "testset/"

# Create an index of class names

class_names = sorted(os.listdir(all_dir))

"""
TODO: Saved inception model to h5 and load it here
"""
model1 = load_model('mobilenet2.h5')
model2 = load_model('InceptionV34.h5')
model3 = load_model('densenet1.h5')

models = [model1, model2, model3]

def ensemble_predictions(members, testX):
    	# make predictions
	yhats = [model.predict(testX) for model in members]
	yhats = np.array(yhats)
	# sum across ensemble members
	summed = np.sum(yhats, axis=0)
	# argmax across classes
	result = np.argmax(summed, axis=1)
	return result


with open("submissionEnsemble3.csv", "w") as fp:
    fp.write("Id,Category\n")

    # Image index
    i = 0
    # 1. load image and resize
    for file in os.listdir(test_dir):
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
            class_index = ensemble_predictions(models, img)[0]

            # Convert class id to name
            label = class_names[class_index]

            fp.write("%d,%s\n" % (i, label))

            print(i)
            i += 1
