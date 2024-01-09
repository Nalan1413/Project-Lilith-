# No Truce With The Furies
import keras_nlp
import keras
import os

keras.mixed_precision.set_global_policy("mixed_float16")

BATCH_SIZE = 16
imdb_train = keras.utils.text_dataset_from_directory(
    "/Users/nalansuo/Downloads/aclImdb/train",
    batch_size=BATCH_SIZE,
)
imdb_test = keras.utils.text_dataset_from_directory(
    "/Users/nalansuo/Downloads/aclImdb/test",
    batch_size=BATCH_SIZE,
)

print(imdb_train.unbatch().take(1).get_single_element())
# ...
