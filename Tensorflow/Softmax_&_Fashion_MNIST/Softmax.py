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
print("\nPrediction for -%s-" % pic)
predictions = model.predict(image_array)
max_index = np.argmax(predictions)

if max_index == 0:
    print("It is a T-shirt/top")
if max_index == 1:
    print("It is a Trouser")
if max_index == 2:
    print("It is a Pullover")
if max_index == 3:
    print("It is a Dress")
if max_index == 4:
    print("It is a Coat")
if max_index == 5:
    print("It is a Sandal")
if max_index == 6:
    print("It is a Shirt")
if max_index == 7:
    print("It is a Sneaker")
if max_index == 8:
    print("It is a Bag")
if max_index == 9:
    print("It is a Ankle boot")

"""
Epoch 1/20
1875/1875 [==============================] - 1s 559us/step - loss: 0.5009 - acc: 0.8259
Epoch 2/20
1875/1875 [==============================] - 1s 561us/step - loss: 0.3767 - acc: 0.8638
Epoch 3/20
1875/1875 [==============================] - 1s 533us/step - loss: 0.3362 - acc: 0.8778
Epoch 4/20
1875/1875 [==============================] - 1s 526us/step - loss: 0.3123 - acc: 0.8855
Epoch 5/20
1875/1875 [==============================] - 1s 568us/step - loss: 0.2945 - acc: 0.8917
Epoch 6/20
1875/1875 [==============================] - 1s 562us/step - loss: 0.2800 - acc: 0.8966
Epoch 7/20
1875/1875 [==============================] - 1s 563us/step - loss: 0.2677 - acc: 0.9003
Epoch 8/20
1875/1875 [==============================] - 1s 753us/step - loss: 0.2565 - acc: 0.9041
Epoch 9/20
1875/1875 [==============================] - 1s 732us/step - loss: 0.2487 - acc: 0.9080
Epoch 10/20
1875/1875 [==============================] - 1s 739us/step - loss: 0.2388 - acc: 0.9108
Epoch 11/20
1875/1875 [==============================] - 2s 812us/step - loss: 0.2320 - acc: 0.9136
Epoch 12/20
1875/1875 [==============================] - 1s 656us/step - loss: 0.2234 - acc: 0.9173
Epoch 13/20
1875/1875 [==============================] - 1s 657us/step - loss: 0.2183 - acc: 0.9182
Epoch 14/20
1875/1875 [==============================] - 1s 733us/step - loss: 0.2138 - acc: 0.9189
Epoch 15/20
1875/1875 [==============================] - 1s 763us/step - loss: 0.2040 - acc: 0.9234
Epoch 16/20
1875/1875 [==============================] - 1s 693us/step - loss: 0.2009 - acc: 0.9251
Epoch 17/20
1875/1875 [==============================] - 1s 699us/step - loss: 0.1924 - acc: 0.9269
Epoch 18/20
1875/1875 [==============================] - 1s 680us/step - loss: 0.1901 - acc: 0.9276
Epoch 19/20
1875/1875 [==============================] - 1s 599us/step - loss: 0.1839 - acc: 0.9320
Epoch 20/20
1875/1875 [==============================] - 1s 580us/step - loss: 0.1807 - acc: 0.9314

Evaluate test data
313/313 [==============================] - 0s 588us/step - loss: 0.3594 - acc: 0.8881

Prediction for -/Users/nalansuo/Desktop/2.jpg-
1/1 [==============================] - 0s 43ms/step
It is a Trouser
"""
