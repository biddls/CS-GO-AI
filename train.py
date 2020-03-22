import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import cv2

path = "C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS-GO-AI\\data\\AI\\"

feattrain = np.load(path + 'FeatureTrain.npy').reshape((64, 600, 800, 1))
feattest = np.load(path + 'FeatureTest.npy').reshape((16, 600, 800, 1))
lbltrain = np.load(path + 'LableTrain.npy').reshape((64, 90, 1))
lbltest = np.load(path + 'LabelTest.npy').reshape(16, 90, 1)

model = models.Sequential()
model.add(layers.Conv2D(64, (7, 7),strides= 2, activation='relu', input_shape=(feattrain[0].shape)))
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
model.add(layers.Dense(90))


print(lbltest.shape)
print(lbltest)

print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=['MSE'])

history = model.fit(feattrain, lbltrain, epochs=1, batch_size= int(len(feattrain)/8), validation_data=(feattest, lbltest), verbose= 1)

pred = model.predict(feattest)
print(pred)
#this is to draw bounding boxes around stuff
for index, x in enumerate(pred):
    image = cv2.cvtColor(feattest[index], cv2.COLOR_GRAY2RGB)
    for y in range(9):

        centerx = x[2] * 800
        centery = x[3] * 600
        width = x[4] * 800
        height = x[5] * 600

        topleft = (int(centerx - width/2), int(centery - height/2))
        bottomright = (int(centerx + width/2), int(centery + height/2))

        image = cv2.rectangle(image, topleft, bottomright, (255, 0, 0), 2)
        image = cv2.putText(image, x[0], (int(centerx - width/2), int(centery - height/2)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

        x = x[10:]

sleep(1)
cv2.imshow('window', image)
cv2.waitKey()

#cv2.cvtColor(c, cv2.COLOR_BGR2RGB)