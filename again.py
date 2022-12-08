from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
img = cv2.imread("46.jpg")
new_model = load_model("models\imageclassifer.h5")

plt.imshow(img)
plt.show()

resize = tf.image.resize(img,[256,256])
plt.imshow(resize.numpy().astype(int))
plt.show()

y = new_model.predict(np.expand_dims(resize/255,0))

if y> 0.5:
    print("sad")
else:
    print("happy")
    