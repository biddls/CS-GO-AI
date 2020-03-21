import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import glob, os
import pandas as pd

path = "C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS-GO-AI\\data\\AI\\"

feattrain = np.load(path + 'FeatureTrain.npy')
feattest = np.load(path + 'FeatureTest.npy')
lbltrain = np.load(path + 'LableTrain.npy')
lbltest = np.load(path + 'LabelTest.npy')

model = models.Sequential()
model.add(layers.Conv2D(64, (7, 7),strides= 2, activation='relu', input_shape=(600, 800, 1)))
model.add(layers.MaxPooling2D(pool_size= (2, 2), strides= 2))
model.add(layers.Conv2D(64,   (3, 3), strides= 1, activation='relu'))
model.add(layers.Conv2D(64,   (3, 3), strides= 1, activation='linear'))
model.add(layers.Conv2D(64,   (3, 3), strides= 1, activation='relu'))
model.add(layers.Conv2D(64,   (3, 3), strides= 1, activation='linear'))
model.add(layers.Conv2D(128,  (3, 3), strides= 2, activation='relu'))
model.add(layers.Conv2D(128,  (3, 3), strides= 1, activation='relu'))
model.add(layers.Conv2D(128,  (3, 3), strides= 1, activation='linear'))
model.add(layers.Conv2D(256,  (3, 3), strides= 1, activation='relu'))
model.add(layers.Conv2D(256,  (3, 3), strides= 1, activation='linear'))
model.add(layers.Conv2D(512,  (3, 3), strides= 2, activation='relu'))
model.add(layers.Conv2D(512,  (3, 3), strides= 1, activation='linear'))
model.add(layers.Conv2D(512,  (3, 3), strides= 1, activation='linear'))
model.add(layers.Conv2D(1000, (1, 1), strides= 1, activation='linear'))
model.add(layers.Flatten())
model.add(layers.Dense(256))
model.add(layers.Dense(128))
model.add(layers.Dense(100))


print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=['MSE'])

history = model.fit(feattrain, lbltrain, epochs=10,
                    validation_data=(feattest, lbltest))