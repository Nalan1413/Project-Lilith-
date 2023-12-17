# No Truce With The Furies
# This a code that helps you generate pictures of your own
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def ImageDone(path):
    original_image = Image.open(path)
    grayscale_image = original_image.convert("L")
    inverted_image = Image.eval(grayscale_image, lambda x: 255 - x)
    resized_image = inverted_image.resize((28, 28))
    image_array = np.array(resized_image)
    image_batch = np.expand_dims(image_array, axis=0)
    return image_batch
    
