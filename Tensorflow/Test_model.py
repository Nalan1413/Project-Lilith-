# No Truce With The Furies
import tensorflow as tf
import backup_image as bi
import numpy as np

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
loaded_model = tf.saved_model.load("./Saved_model")
url = "https://upload.wikimedia.org/wikipedia/commons/4/40/Sunflower_sky_backdrop.jpg"
img_height = 180
img_width = 180
image_array = bi.package_image(url, img_height, img_width)
predictions = loaded_model(image_array)
ans_index = np.argmax(predictions)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[ans_index], 100 * np.max(predictions[0]))
)
