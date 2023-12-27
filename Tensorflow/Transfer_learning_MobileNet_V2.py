# No Truce With The Furies
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file("cat_and_dog.zip", origin=_URL, extract=True)
# print(path_to_zip)
# /Users/nalansuo/.keras/datasets/cat_and_dog.zip
PATH = os.path.join(os.path.dirname(path_to_zip), "cats_and_dogs_filtered")
train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")
a = os.path.getsize(train_dir)
print(a)
batch_size = 32
img_size = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir, shuffle=True,
                                                            batch_size=batch_size, image_size=img_size)
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir, shuffle=True,
                                                                 batch_size=batch_size, image_size=img_size)
class_names = train_dataset.class_names
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
# plt.show()

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.models.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(160, 160, 3)),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomRotation(0.2)
])

# Example
plt.figure(figsize=(10, 10))
for image, _ in train_dataset.take(1):
    first_image = image[0]
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis("off")
# plt.show()

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

rescale = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)

# base model MobileNet V2
imge_shape = img_size + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=imge_shape, include_top=False, weights="imagenet")
base_model.trainable = False

image_batch, label_batch = next(iter(train_dataset))
image_batch = data_augmentation(image_batch)
feature_batch = base_model(image_batch)
print(feature_batch.shape)
# (32, 5, 5, 1280)

# Pooling
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# Raw prediction
prediction_layer = tf.keras.layers.Dense(1)

# Modeling
inputs = tf.keras.Input(shape=imge_shape)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')])

history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)

# graph
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="training accuracy")
plt.plot(val_acc, label="validation accuracy")
plt.legend(loc="lower right")
plt.ylabel("accuracy")
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')
plt.xlabel("epoch")

plt.subplot(2, 1, 2)
plt.plot(loss, label="train loss")
plt.plot(val_loss, label="validation loss")
plt.legend(loc="upper right")
plt.ylabel("loss")
plt.ylim([min(plt.ylim()), 1])
plt.title("training and validation loss")
plt.xlabel("epoch")
# plt.show()

# unfreeze
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001 / 10),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')])

history_fine = model.fit(train_dataset,
                         validation_data=validation_dataset,
                         epochs=20,
                         initial_epoch=history.epoch[-1])

# Graphing the last part
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label="training accuracy")
plt.plot(val_acc, label="validation accuracy")
plt.ylim([0.8, 1])
plt.plot([9, 9],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc="lower right")
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label="training loss")
plt.plot(val_loss, label="validation loss")
plt.ylim([0.8, 1])
plt.plot([9, 9],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc="upper right")
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.show()

# Test
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")
