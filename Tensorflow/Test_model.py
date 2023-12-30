# No Truce With The Furies
import tensorflow as tf
import backup_image as bi
import numpy as np

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
loaded_model = tf.saved_model.load("./Saved_model")
url = "https://www.thespruce.com/thmb/tSi9I62mHGnX9DbORpErX4Mjc0Y=/3000x0/filters:no_upscale():max_bytes(150000):strip_icc()/no-respect-for-clover-and-dandelion-weeds-2153155-hero-e5fe10c1d40b41a68219c9705f6e3b88.jpg"
filename = "1.jpg"
img_height = 180
img_width = 180
image_array = bi.package_image(url, filename, img_height, img_width)
predictions = loaded_model(image_array)
ans_index = np.argmax(predictions)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[ans_index], 100 * np.max(predictions[0]))
)
