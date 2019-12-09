# Load all necessary modules
import os
import tensorflow as tf 
import cv2
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import scipy.io as sio 
# Create an index of class names

class_names = sorted(os.listdir(r"D:\Downloads\DOCUMENTS\STUDIES\ml\pattern_recognition_ml\Code\project\train\train"))
# Prepare a pretrained CNN for feature extraction

base_model = tf.keras.applications.mobilenet.MobileNet(
    input_shape = (224, 224, 3), 
    include_top = False)


# Get the network structure
print(base_model.summary())

#
in_tensor = base_model.inputs[0]
out_tensor = base_model.outputs[0]

out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)

# Define the full model by the endpoints
model = tf.keras.models.Model(inputs = [in_tensor], outputs = [out_tensor])

# Compile the model for execution. Losses and optimizers can be 
# anything here, since we don't train the model
model.compile(loss= "categorical_crossentropy", optimizer= 'sgd')


X = []
y = []

for root, dirs, files in os.walk(r"D:\Downloads\DOCUMENTS\STUDIES\ml\pattern_recognition_ml\Code\project\train\train"):
    for name in files:
        if name.endswith(".jpg"):

            # Load the image
            img = plt.imread(root + os.sep + name)

            # Resize it to the net input size:
            img = cv2.resize(img, (224, 224))

            # Convert the data to float, and remove mean:
            img = img.astype(np.float32)
            img -= 128

            # Push the data through the mode:
            x = model.predict(img[np.newaxis, ...])[0] # turn each point in the matrix into 1 input point

            # And append the feature vector to our list
            X.append(x)

            name = os.path.join(root, name)
            
            label = name.split(os.sep)[-2]
            print(label)
            y.append(class_names.index(label))


# Cast the python lists to a numpy array
X = np.array(X)
y = np.array(y)

print(X.shape)
print(y.shape)

sio.savemat('features.mat', mdict={'X' : X, 'y' : y})

# Split images into train and validation folder 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size = 0.8)

# Linear discriminant analysis classifier
lda = LinearDiscriminantAnalysis()
SVC_linear = SVC(kernel='linear')
SVC_rbf = SVC(kernel='rbf')
logistic_regression = LogisticRegression()
rand_forest = RandomForestClassifier(n_estimators=100)

classifiers = [lda, SVC_linear, SVC_rbf, logistic_regression, rand_forest]

for classifier in classifiers:
    classifier.fit(X_train, y_train)
    predict = classifier.predict(X_test)
    print(accuracy_score(y_test, predict))







