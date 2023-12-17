# No Truce With The Furies
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from backup_image import ImageDone


(train_image, train_lable), (test_image, test_lable) = tf.keras.datasets.fashion_mnist.load_data()
# print(train_image.shape)  # (60000, 28, 28)
# print(train_lable.shape)  # (60000,)
# print(test_image.shape, test_lable.shape)  # (10000, 28, 28) (10000,)

plt.imshow(train_image[0])
# plt.show()

# Making the datasets into numbers smaller than one
train_image = train_image / 255
test_image = test_image / 255

# Setting up the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

# Training
model.fit(train_image, train_lable, epochs=9)

# Testing
print("\nEvaluate test data")
model.evaluate(test_image, test_lable)

# Getting some image yourself
pic = "./shoe.webp"  # The path to your own image
Own_image = ImageDone(pic)
image_array = Own_image.grayscale_image_array()
Own_image.show_image()

# Getting prediction
predictions = model.predict(image_array)
print(predictions)


"""
Epoch 1/9
1875/1875 [==============================] - 1s 539us/step - loss: 0.4953 - acc: 0.8260
Epoch 2/9
1875/1875 [==============================] - 1s 525us/step - loss: 0.3747 - acc: 0.8645
Epoch 3/9
1875/1875 [==============================] - 1s 540us/step - loss: 0.3384 - acc: 0.8777
Epoch 4/9
1875/1875 [==============================] - 1s 525us/step - loss: 0.3151 - acc: 0.8841
Epoch 5/9
1875/1875 [==============================] - 1s 538us/step - loss: 0.2974 - acc: 0.8906
Epoch 6/9
1875/1875 [==============================] - 1s 559us/step - loss: 0.2814 - acc: 0.8956
Epoch 7/9
1875/1875 [==============================] - 1s 539us/step - loss: 0.2707 - acc: 0.9008
Epoch 8/9
1875/1875 [==============================] - 1s 531us/step - loss: 0.2587 - acc: 0.9040
Epoch 9/9
1875/1875 [==============================] - 1s 532us/step - loss: 0.2473 - acc: 0.9061

Evaluate test data
313/313 [==============================] - 0s 377us/step - loss: 0.3324 - acc: 0.8825
1/1 [==============================] - 0s 46ms/step
[[0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]]
"""
