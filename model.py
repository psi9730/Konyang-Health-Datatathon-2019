import os
import argparse
import sys
import time
import random
import cv2
import numpy as np
import keras

from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D
from keras.layers import BatchNormalization, ReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import  ModelCheckpoint, ReduceLROnPlateau
import keras.backend.tensorflow_backend as K
import nsml
from nsml import DATASET_PATH, GPU_NUM



def cnn_sample(in_shape, num_classes=4):    # Example CNN
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', input_shape=in_shape))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(Conv2D(filters=32, kernel_size=3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=96, kernel_size=3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(Conv2D(filters=96, kernel_size=3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add((ReLU()))
    model.add(Conv2D(filters=256, kernel_size=3, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add((ReLU()))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    return model

