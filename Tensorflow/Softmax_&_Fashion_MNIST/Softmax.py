# No Truce With The Furies
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import backup_image as bi


(train_image, train_lable), (test_image, test_lable) = tf.keras.datasets.fashion_mnist.load_data()
# print(train_image.shape)  # (60000, 28, 28)
# print(train_lable.shape)  # (60000,)
# print(test_image.shape, test_lable.shape)  # (10000, 28, 28) (10000,)

plt.imshow(test_image[0])
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
model.fit(train_image, train_lable, epochs=20)

# Testing
print("\nEvaluate test data")
model.evaluate(test_image, test_lable)

# Getting some image yourself
pic = "./trousers.jpg"  # The path to your own image
image_array = bi.ImageDone(pic)

plt.imshow(image_array[0])
# plt.show()

image_array = image_array / 255

# Getting prediction
predictions = model.predict(image_array)
print(predictions)

"""
Epoch 1/20
1875/1875 [==============================] - 1s 532us/step - loss: 0.4969 - acc: 0.8246
Epoch 2/20
1875/1875 [==============================] - 1s 544us/step - loss: 0.3757 - acc: 0.8655
Epoch 3/20
1875/1875 [==============================] - 1s 553us/step - loss: 0.3381 - acc: 0.8769
Epoch 4/20
1875/1875 [==============================] - 1s 519us/step - loss: 0.3137 - acc: 0.8856
Epoch 5/20
1875/1875 [==============================] - 1s 517us/step - loss: 0.2946 - acc: 0.8911
Epoch 6/20
1875/1875 [==============================] - 1s 527us/step - loss: 0.2803 - acc: 0.8956
Epoch 7/20
1875/1875 [==============================] - 1s 531us/step - loss: 0.2665 - acc: 0.9001
Epoch 8/20
1875/1875 [==============================] - 1s 531us/step - loss: 0.2564 - acc: 0.9041
Epoch 9/20
1875/1875 [==============================] - 1s 522us/step - loss: 0.2482 - acc: 0.9087
Epoch 10/20
1875/1875 [==============================] - 1s 540us/step - loss: 0.2384 - acc: 0.9116
Epoch 11/20
1875/1875 [==============================] - 1s 595us/step - loss: 0.2299 - acc: 0.9139
Epoch 12/20
1875/1875 [==============================] - 1s 559us/step - loss: 0.2244 - acc: 0.9166
Epoch 13/20
1875/1875 [==============================] - 1s 521us/step - loss: 0.2157 - acc: 0.9201
Epoch 14/20
1875/1875 [==============================] - 1s 539us/step - loss: 0.2112 - acc: 0.9208
Epoch 15/20
1875/1875 [==============================] - 1s 543us/step - loss: 0.2049 - acc: 0.9228
Epoch 16/20
1875/1875 [==============================] - 1s 542us/step - loss: 0.1980 - acc: 0.9265
Epoch 17/20
1875/1875 [==============================] - 1s 548us/step - loss: 0.1931 - acc: 0.9269
Epoch 18/20
1875/1875 [==============================] - 1s 515us/step - loss: 0.1878 - acc: 0.9301
Epoch 19/20
1875/1875 [==============================] - 1s 530us/step - loss: 0.1835 - acc: 0.9303
Epoch 20/20
1875/1875 [==============================] - 1s 548us/step - loss: 0.1775 - acc: 0.9327

Evaluate test data
313/313 [==============================] - 0s 376us/step - loss: 0.3527 - acc: 0.8862
1/1 [==============================] - 0s 39ms/step
[[8.3488274e-05 9.4328487e-01 3.8799706e-05 5.5775803e-02 4.2407072e-04
  9.4059552e-12 3.9282476e-04 7.3972444e-14 7.9311455e-08 1.8289868e-12]]
"""

