# No Truce With The Furies
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
data_dir = pathlib.Path(data_dir).with_suffix('')

directory_names = [directory.name for directory in data_dir.glob("*/")][:-1]
# print(directory_names)
# ['roses', 'sunflowers', 'daisy', 'dandelion', 'tulips']

image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)
# 3670

rose = list(data_dir.glob("roses/*"))

# Creating a dataset
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=123, image_size=(
        img_height, img_width), batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=123,
    image_size=(img_height, img_width), batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

# Showing the pictures
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
# plt.show()

# Configure datasets to improve performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# Setting up
model = tf.keras.models.Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation="softmax")
])

# Train
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Ploting graph
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epoch_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epoch_range, acc, label="Training Accuracy")
plt.plot(epoch_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epoch_range, loss, label="Training Loss")
plt.plot(epoch_range, val_loss, label="Validation Loass")
plt.legend(loc="upper right")
plt.title('Training and Validation Loss')
plt.show()

# Predict
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
predictions = model.predict(img_array)

ans_index = np.argmax(predictions)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[ans_index], 100 * np.max(predictions[0]))
)


"""
Found 3670 files belonging to 5 classes.
Using 2936 files for training.
Found 3670 files belonging to 5 classes.
Using 734 files for validation.
['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
Epoch 1/15
92/92 [==============================] - 19s 190ms/step - loss: 1.3033 - accuracy: 0.4530 - val_loss: 1.1201 - val_accuracy: 0.5490
Epoch 2/15
92/92 [==============================] - 18s 190ms/step - loss: 1.0580 - accuracy: 0.5834 - val_loss: 1.0157 - val_accuracy: 0.5967
Epoch 3/15
92/92 [==============================] - 18s 197ms/step - loss: 0.9677 - accuracy: 0.6189 - val_loss: 1.1114 - val_accuracy: 0.6281
Epoch 4/15
92/92 [==============================] - 19s 210ms/step - loss: 0.8862 - accuracy: 0.6628 - val_loss: 0.8974 - val_accuracy: 0.6485
Epoch 5/15
92/92 [==============================] - 19s 207ms/step - loss: 0.8330 - accuracy: 0.6771 - val_loss: 0.8114 - val_accuracy: 0.6853
Epoch 6/15
92/92 [==============================] - 19s 208ms/step - loss: 0.7676 - accuracy: 0.7163 - val_loss: 0.7925 - val_accuracy: 0.6962
Epoch 7/15
92/92 [==============================] - 19s 212ms/step - loss: 0.7241 - accuracy: 0.7279 - val_loss: 0.7965 - val_accuracy: 0.6785
Epoch 8/15
92/92 [==============================] - 19s 211ms/step - loss: 0.7074 - accuracy: 0.7302 - val_loss: 0.7623 - val_accuracy: 0.7125
Epoch 9/15
92/92 [==============================] - 21s 230ms/step - loss: 0.6762 - accuracy: 0.7490 - val_loss: 0.7085 - val_accuracy: 0.7302
Epoch 10/15
92/92 [==============================] - 20s 221ms/step - loss: 0.6303 - accuracy: 0.7558 - val_loss: 0.7997 - val_accuracy: 0.6975
Epoch 11/15
92/92 [==============================] - 20s 220ms/step - loss: 0.6427 - accuracy: 0.7558 - val_loss: 0.7125 - val_accuracy: 0.7071
Epoch 12/15
92/92 [==============================] - 20s 222ms/step - loss: 0.5940 - accuracy: 0.7742 - val_loss: 0.7008 - val_accuracy: 0.7207
Epoch 13/15
92/92 [==============================] - 22s 239ms/step - loss: 0.5784 - accuracy: 0.7728 - val_loss: 0.7492 - val_accuracy: 0.7139
Epoch 14/15
92/92 [==============================] - 21s 229ms/step - loss: 0.5553 - accuracy: 0.7922 - val_loss: 0.6608 - val_accuracy: 0.7384
Epoch 15/15
92/92 [==============================] - 21s 226ms/step - loss: 0.5265 - accuracy: 0.7994 - val_loss: 0.6929 - val_accuracy: 0.7384
1/1 [==============================] - 0s 102ms/step
This image most likely belongs to sunflowers with a 99.59 percent confidence.
"""
