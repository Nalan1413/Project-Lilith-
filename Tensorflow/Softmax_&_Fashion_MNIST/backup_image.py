# No Truce With The Furies
# This a code that helps you generate pictures of your own
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class ImageDone:
    def __init__(self, path):
        self.original_image = Image.open(path)
        grayscale_image = self.original_image.convert("L")
        self.resized_image = grayscale_image.resize((28, 28))
        self.image_array = np.array(self.resized_image)

    # Getting the gray image batch
    def grayscale_image_array(self):
        image_batch = np.expand_dims(self.image_array, axis=0)
        return image_batch

    # Showing the image in 28 * 28 formation
    def show_image(self):
        plt.imshow(self.image_array)
        plt.show()
