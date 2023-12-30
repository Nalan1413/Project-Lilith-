# No Truce With The Furies
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def ImageDone(path):
    original_image = Image.open(path)
    grayscale_image = original_image.convert("L")
    inverted_image = Image.eval(grayscale_image, lambda x: 255 - x)
    resized_image = inverted_image.resize((180, 180))
    image_array = np.array(resized_image)
    image_array = image_array / 255
    image_batch = np.expand_dims(image_array, axis=0)
    return image_batch


def package_image(url, fname, img_height, img_width):
    image_file = tf.keras.utils.get_file(fname, origin=url)
    img = tf.keras.utils.load_img(
        image_file, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    os.remove(image_file)
    return img_array
