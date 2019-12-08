import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from helpers import separate_test_train_dirs, generate_augment
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.externals import joblib
from tensorflow.keras.models import Model

input_generator_shape = (224, 224)
input_shape = (224, 224, 3) # including color channels
batch_size = 32

# All paths needs a "/" at the end for this to work
all_path = r"C:\Users\RogerRoger\PycharmProjects\ML\Vehicle-Type-Detection-master\train\train/" # CHANGE THIS. directory path to where all images are
train_path = r"C:\Users\RogerRoger\PycharmProjects\ML\Vehicle-Type-Detection-master\Part2\Vehicle-Type-Detection/train_only/" # directory path to store train images
test_path = r"C:\Users\RogerRoger\PycharmProjects\ML\Vehicle-Type-Detection-master\Part2\Vehicle-Type-Detection/test_only/" # directory path to store test images

# Separate images into train and testR
separate_test_train_dirs(all_path, train_path, test_path)

num_classes = 17 # number of classes in the data
epochs = 12


base_model = tf.keras.applications.inception_v3.InceptionV3(
    input_shape = input_shape,
    include_top = False, pooling = "avg")

print(base_model.summary())

in_tensor = base_model.input
out_tensor = base_model.output

w = base_model.output
#w = Flatten(w)
w = GlobalAveragePooling2D()(w)
w = Dense(128, activation = "relu")(w)
output = Dense(1, activation = "softmax")(w)
model = Model(inputs = [base_model.input], outputs = [output])


model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()

# Get generators for training and validation
train_generator, validation_generator = generate_augment(train_path, 
										test_path,
										input_generator_shape,
										batch_size)

model.fit_generator(train_generator,
				    steps_per_epoch = train_generator.samples // batch_size,
				    validation_data = validation_generator, 
				    validation_steps = validation_generator.samples // batch_size,
					epochs=epochs)

filename = 'trained_InceptionV3'
joblib.dump(model, filename)