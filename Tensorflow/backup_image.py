# No Truce With The Furies

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def ImageDone(path):
    original_image = Image.open(path)
    grayscale_image = original_image.convert("L")
    inverted_image = Image.eval(grayscale_image, lambda x: 255 - x)
    resized_image = inverted_image.resize((28, 28))
    image_array = np.array(resized_image)
    image_batch = np.expand_dims(image_array, axis=0)
    return image_batch


def package_image(url, img_height, img_width):
    image_file = tf.keras.utils.get_file('flower.jpg', origin=url)
    img = tf.keras.utils.load_img(
        image_file, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array
