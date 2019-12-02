import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from helpers import separate_test_train_dirs, generate_augment

input_generator_shape = (224, 224)
input_shape = (224, 224, 3) # including color channels
batch_size = 32

# All paths needs a "/" at the end for this to work
all_path = "./all_images/" # CHANGE THIS. directory path to where all images are
train_path = "./train_only/" # directory path to store train images
test_path = "./test_only/" # directory path to store test images

# Separate images into train and test
separate_test_train_dirs(all_path, train_path, test_path)

num_classes = 17 # number of classes in the data
epochs = 12

### A simple neural network from the exercises
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adadelta(),
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