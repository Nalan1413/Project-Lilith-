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

# plt.imshow(test_image[0])
# plt.show()

# Making the datasets into numbers smaller than one
train_image = train_image / 255
test_image = test_image / 255

# Setting up the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

# Training and Testing
history = model.fit(train_image, train_lable, epochs=18, validation_data=(test_image, test_lable))
# Or use "model.evaluate(test_image, test_lable)" to test the reslts instead of validation

# Plot graph of training
# plt.plot(history.epoch, history.history.get("loss"), label="loss")
# plt.plot(history.epoch, history.history.get("val_loss"), label="val_loss")
# plt.legend()
# plt.show()

# Getting some image yourself
pic = "/Users/nalansuo/Desktop/dress.webp"  # The path to your own image
image_array = bi.ImageDone(pic)

plt.imshow(image_array[0])
plt.show()

image_array = image_array / 255

# Getting prediction
print("\nPrediction for -%s-" % pic)
predictions = model.predict(image_array)
max_index = np .argmax(predictions)

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
Epoch 1/18
1875/1875 [==============================] - 2s 810us/step - loss: 0.5960 - acc: 0.7865 - val_loss: 0.4546 - val_acc: 0.8369
Epoch 2/18
1875/1875 [==============================] - 1s 786us/step - loss: 0.4519 - acc: 0.8357 - val_loss: 0.3997 - val_acc: 0.8536
Epoch 3/18
1875/1875 [==============================] - 1s 772us/step - loss: 0.4118 - acc: 0.8495 - val_loss: 0.3834 - val_acc: 0.8569
Epoch 4/18
1875/1875 [==============================] - 1s 768us/step - loss: 0.3909 - acc: 0.8570 - val_loss: 0.3789 - val_acc: 0.8597
Epoch 5/18
1875/1875 [==============================] - 1s 765us/step - loss: 0.3730 - acc: 0.8629 - val_loss: 0.3748 - val_acc: 0.8627
Epoch 6/18
1875/1875 [==============================] - 1s 760us/step - loss: 0.3662 - acc: 0.8654 - val_loss: 0.3577 - val_acc: 0.8711
Epoch 7/18
1875/1875 [==============================] - 1s 750us/step - loss: 0.3540 - acc: 0.8691 - val_loss: 0.3562 - val_acc: 0.8738
Epoch 8/18
1875/1875 [==============================] - 1s 785us/step - loss: 0.3461 - acc: 0.8726 - val_loss: 0.3480 - val_acc: 0.8751
Epoch 9/18
1875/1875 [==============================] - 1s 764us/step - loss: 0.3402 - acc: 0.8749 - val_loss: 0.3532 - val_acc: 0.8714
Epoch 10/18
1875/1875 [==============================] - 1s 768us/step - loss: 0.3334 - acc: 0.8777 - val_loss: 0.3490 - val_acc: 0.8743
Epoch 11/18
1875/1875 [==============================] - 1s 769us/step - loss: 0.3263 - acc: 0.8799 - val_loss: 0.3552 - val_acc: 0.8704
Epoch 12/18
1875/1875 [==============================] - 1s 771us/step - loss: 0.3212 - acc: 0.8806 - val_loss: 0.3448 - val_acc: 0.8749
Epoch 13/18
1875/1875 [==============================] - 1s 770us/step - loss: 0.3179 - acc: 0.8834 - val_loss: 0.3434 - val_acc: 0.8818
Epoch 14/18
1875/1875 [==============================] - 1s 763us/step - loss: 0.3153 - acc: 0.8831 - val_loss: 0.3322 - val_acc: 0.8824
Epoch 15/18
1875/1875 [==============================] - 1s 788us/step - loss: 0.3099 - acc: 0.8855 - val_loss: 0.3368 - val_acc: 0.8804
Epoch 16/18
1875/1875 [==============================] - 1s 769us/step - loss: 0.3071 - acc: 0.8859 - val_loss: 0.3458 - val_acc: 0.8782
Epoch 17/18
1875/1875 [==============================] - 1s 762us/step - loss: 0.3055 - acc: 0.8877 - val_loss: 0.3447 - val_acc: 0.8740
Epoch 18/18
1875/1875 [==============================] - 1s 763us/step - loss: 0.3044 - acc: 0.8867 - val_loss: 0.3237 - val_acc: 0.8863

Prediction for -/Users/nalansuo/Desktop/dress.webp-
1/1 [==============================] - 0s 44ms/step
It is a Dress
"""
