import os
import shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator 

def separate_test_train_dirs(all_data_dir, training_data_dir, testing_data_dir, testing_data_pct = 0.2):
    """
    Code snippet from github.com/daanraman
    
    Separate all images into test and train directories. It copies images from a directory 
    of pictures in subdirectories (classes) to 2 separate directories: train and test, both
    have the same class subdirectories. 
    """
    # Recreate testing and training directories
    if testing_data_dir.count('/') > 1:
        shutil.rmtree(testing_data_dir, ignore_errors=False)
        os.makedirs(testing_data_dir)
        print("Successfully cleaned directory " + testing_data_dir)
    else:
        print("Refusing to delete testing data directory " + testing_data_dir + " as we prevent you from doing stupid things!")

    if training_data_dir.count('/') > 1:
        shutil.rmtree(training_data_dir, ignore_errors=False)
        os.makedirs(training_data_dir)
        print("Successfully cleaned directory " + training_data_dir)
    else:
        print("Refusing to delete testing data directory " + training_data_dir + " as we prevent you from doing stupid things!")

    num_training_files = 0
    num_testing_files = 0

    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = os.path.basename(subdir)

        # Don't create a subdirectory for the root directory
        print(category_name + " vs " + os.path.basename(all_data_dir))
        if category_name == os.path.basename(all_data_dir):
            continue

        training_data_category_dir = training_data_dir + '/' + category_name
        testing_data_category_dir = testing_data_dir + '/' + category_name

        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)

        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)

        for file in files:
            input_file = os.path.join(subdir, file)
            if np.random.rand(1) < testing_data_pct:
                shutil.copy(input_file, testing_data_dir + '/' + category_name + '/' + file)
                num_testing_files += 1
            else:
                shutil.copy(input_file, training_data_dir + '/' + category_name + '/' + file)
                num_training_files += 1

    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_testing_files) + " testing files.")


def generate_augment(train_path, test_path, shape, batch_size = 32):
    """
    Generate images & do some augmentations

    @params: path: directory path to where the images are
             batch_size: size of each batch (default 32)
             shape: input shape of images (width and height only)
    """
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True, 
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2)

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=shape,
        batch_size=batch_size)

    print(train_generator.class_indices)

    validation_generator = test_datagen.flow_from_directory(
        test_path, # same directory as training data
        target_size=shape,
        batch_size=batch_size)

    return train_generator, validation_generator
