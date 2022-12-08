import tensorflow as tf
import os
import cv2
import imghdr

image_ext= ['jpeg','jpg','bmp','png']

import numpy as np
import matplotlib.pyplot as plt
data = tf.keras.utils.image_dataset_from_directory("dataset/training_set")
data = data.map(lambda x,y:(x/255.0,y))


data_iterator = data.as_numpy_iterator()
data = data_iterator.next()



