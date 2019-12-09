import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model


"""
    Combine the result of models together
    @param members: vector of models
    @param test_image: image to predict
    @return the predicted result combined
"""
def ensemble_predictions(members, test_image):
    preds = [model.predict(test_image) for model in members]
    preds = np.array[preds]

    # Sum the result across emsemble
    summed = np.sum(preds, axis=0)
    result = np.argmax(summed, axis=1)

    return result





