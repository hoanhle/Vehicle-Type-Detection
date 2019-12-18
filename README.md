# Vehicle Type Detection
This repository details how our team developed different machine learning models using [scikit-learn](http://scikit-learn.org) and [Keras](https://www.tensorflow.org/guide/keras/overview) to classify images into 16 different types of vehicle (and one extra class Caterpillar). Our project was developed to compete in [TAU Vehicle Type Recognition Competition on Kaggle](https://www.kaggle.com/c/vehicle), a part of the course [Pattern Recognition and Machine Learning 2019](http://www.cs.tut.fi/courses/SGN-41007/) at Tampere University.

## Table of Contents
- [Repository structure](#repository-structure)
- [Competition overview](#competition-overview)
- [Data](#data)
- [Models](#models)
- [Training methods](#training-methods)
- [Accuracy](#accuracy)
- [Further development](#further-development)
- [Keywords](#keywords)

## Repository structure
```
.
├── __pycache__             
├── saved_model             --> pretrained CNN models from Keras that were
│				trained on our dataset (with augmentations)
├── test_linear_model       --> scripts to train different non-CNN
│				models on the train set          
├── train_linear_model      --> cripts to test different non-CNN 
│				models on the test set, which generate .csv submission files           
├── .gitattributes          
├── .gitignore
├── extract_features.py     --> extract image features using CNN (MobileNet)
├── features.zip            --> images features extracted using MobileNet
├── finalSubmission.csv     --> final Kaggle submission file
├── generator_example.py    --> an example how to generate training images in 
│				to optimize memory efficiency
├── helpers.py              --> functions to augment images & generate training
│				images in batches
├── mobileNetV2.py          --> an example how to load a pretrained CNN model
│				(mobileNetV2), train it using your own training data, and save it
├── test_trained_CNN.py     --> test CNN model's accuracy on the test dataset, which
│				generates a .csv file for submission

```

## Competition overview
Please visit the [competition Overview page](https://www.kaggle.com/c/vehicle) for more information

## Data
The data for the competition consists of **training data** together with the class labels and **test data** without the labels. There are a total of 17 classes: Ambulance, Boat, Cart, Limousine, Snowmobile, Truck, Barge, Bus, Caterpillar, Motorcycle, Tank, Van, Bicycle, Car, Helicopter, Segway, Taxi.

The data has been collected from the Open Images dataset; an annotated collection of over 9 million images. We are using a subset of openimages, selected to contain only vehicle categories among the total of 600 object classes.

To download the data, run the command below on the command line:
```
kaggle competitions download -c vehicle
```

The dataset consists of three files listed below.

1. train.zip - the training set: a set of images with true labels in the folder names. The zip file contains altogether 28045 files organized in folders. The folder name is the true class; i.e., "Boat" folder has all boat images, "Car" folder has all the car images and so on.
2. test.zip - the test set: a set of images without labels. The zip file contains altogether 7958 files in a single folder. The file name is the id for the solution's first column; i.e., the predicted class for file "000000.jpg" should appear on the first row of your submission.
3. sample_submission.csv - a sample submission file in the correct format (predicting all "cars" class)

## Training models
We used 5 different common training models: 
* [Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) (CNN) 
* [Support vector machine](https://en.wikipedia.org/wiki/Support-vector_machine) (SVM)
* [Linear discriminant analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) (LDA)
* [Logistic regression model](https://en.wikipedia.org/wiki/Logistic_regression)
* [Random forest](https://en.wikipedia.org/wiki/Random_forest)

Regarding CNN, we trained 4 pretrained models on our own training data: MobileNetV1, MobileNetV2, DenseNet and InceptionV3. Details regarding training methods are explained in [the next section](#training-methods).

## Training methods
For all training models, we perform a 80/20 split on the training data. 80% of the training data will be used for actual training, while the other 20% will be used for validation to make sure the models do not [overfit](https://en.wikipedia.org/wiki/Overfitting).

Among the 5 training models we use, convolutional neural network is proved to be the most resource-intensive, time-consuming, but most accurate model. Therefore, this model is our team's focus when approaching the problem. In order to take full advantage the CNN, our team implemented the following "tricks":
* Feed data on batches (generator): Instead of loading the whole training data into the memory, we will feed it in batches, effectively "generating" training data while the model is being trained. This will optimize memory efficiency as well as let us have greater control over memory usage. File [generator_example.py](https://github.com/hoanhle/Vehicle-Type-Detection/blob/master/generator_example.py) provides an example of doing this.
* Augment training images to prevent overfitting: We perform a few augmentations on the training images, such as shearing, shifting, rotate & flipping. This will distort the original images in a random manner to avoid overfitting, while preserving their features to minimize misclassification. Please note that only train images should be augmented, the validation images should be left as they are. File [helpers.py](https://github.com/hoanhle/Vehicle-Type-Detection/blob/master/helpers.py) details the augmentations.
* [Ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning): the actual classification is a combined decision of MobileNetV2, DenseNet and InceptionV3. We let all three models predict the class for each image and take the majority vote.

## Accuracy
The accuracies of all trained models, tested on the test set are presented in the table below.

| Classifier                              | Validation  accuracy | Kaggle accuracy  (public leaderboard) |
|-----------------------------------------|:--------------------:|:-------------------------------------:|
| Ensembled CNNs                          |                      |                  91%                  |
| MobileNetV1                             |          74%         |                  74%                  |
| MobileNetV2                             |          89%         |                  85%                  |
| DenseNet                                |          89%         |             not submitted             |
| InceptionV3                             |          92%         |                  89%                  |
| Support vector machine  (RBF kernel)    |          78%         |                  75%                  |
| Support vector machine  (Linear kernel) |          73%         |                  71%                  |
| Logistic regression model               |          76%         |                  75%                  |
| Linear discriminant analysis            |          76%         |                  74%                  |
| Random forest                           |          67%         |                  66%                  |

## Further development
The training data provided is very imbalanced. To get higher accuracies, assigning [class weights](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html) during training is desired.

## Keywords
machine learning, deep learning, deep neural network, convolutional neural network, classification, logistic regression, random forest, support vector machine, linear discriminant analysis, tensorflow, keras, sckit-learn.