import tensorflow as tf
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


tf.config.list_physical_devices("GPU")

import cv2
import imghdr


data_dir = 'data'
#check for images extentions
image_exts = ['jpeg', 'jpg','bmp','png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir,image_class)):
        image_path = os.path.join(data_dir,image_class,image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print("Image not found")
                os.remove(image_path)
        except:
            print("issue with image")


import numpy as np
from matplotlib import pyplot as plt

#loading data
data = tf.keras.utils.image_dataset_from_directory("data")
#
data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

fig, ax = plt.subplots(ncols=4,figsize=(20,20))
for idx,img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


data = data.map(lambda x,y:(x/255.0,y))

data.as_numpy_iterator().next()
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

print(train_size)


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32,(3,3),1,activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(16,(3,3),1,activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


    
model.compile('adam',loss=tf.losses.BinaryCrossentropy(),metrics=['accuracy'])

model.summary()


logdir="logs"

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)
hist = model.fit(train,epochs=20,validation_data=val,callbacks=[tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'],color='teal',label='loss')
plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
fig.suptitle("Loss",fontsize=20)
plt.legend(loc="upper right")
plt.show()


fig = plt.figure()
plt.plot(hist.history['accuracy'],color='teal',label='accuracy')
plt.plot(hist.history['val_accuracy'],color='orange',label='val_accuracy')
fig.suptitle("Accuracy",fontsize=20)
plt.legend(loc="upper right")
plt.show()

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()


for batch in test.as_numpy_iterator():
    x,y = batch
    y_pred = model.predict(x)
    pre.update_state(y,y_pred)
    re.update_state(y,y_pred)
    acc.update_state(y,y_pred)

print(pre.result(),re.result(),acc.result())


img = cv2.imread("8iAb9k4aT.jpg")
plt.imshow(img)
plt.show()


resize = tf.image.resize(img,[256,256])
plt.imshow(resize.numpy().astype(int))
plt.show()

y_pre = model.predict(np.expand_dims(resize/255,0))

print(y_pre)

if y_pre > 0.5:
    print(f"predicted class is sad")
else:
    print(f"predicted class is happy")

from tensorflow.keras.models import load_model

model.save(os.path.join("models","imageclassifer.h5"))


