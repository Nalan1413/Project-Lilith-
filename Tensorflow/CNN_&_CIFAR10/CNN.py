# No Truce With The Furies
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Checking the images
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
ax1 = plt.gca()
for i in range(25):
    ax1 = plt.subplot(5, 5, i+1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.grid(False)
    ax1.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    ax1.set_xlabel(class_names[train_labels[i][0]])
# plt.show()

# Setting up the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

# Train
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['acc'])
history = model.fit(train_images, train_labels, epochs=7, validation_data=(test_images, test_labels))

# Ploting results
plt.figure(figsize=(10, 10))
ax2 = plt.gca()
plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label='val_accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
