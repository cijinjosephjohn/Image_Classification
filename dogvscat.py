import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


data = "dataset/training_set"
categories = ["cats","dogs"]

for category in categories:
    path = os.path.join(data,category)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        # plt.imshow(img_arr,cmap="gray")
        # plt.show()
        break
    break

print(img_arr.shape)

#size of image

IMG_SIZE = 75

# new_arr = cv2.resize(img_arr,(IMG_SIZE,IMG_SIZE))
# plt.imshow(new_arr,cmap="gray")
# plt.show()


training_data =[]

def create_training_data():
    for category in categories:
        path = os.path.join(data,category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_arr = cv2.resize(img_arr,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_arr,class_num])
            except Exception as e:
                pass
create_training_data()

# print(len(training_data))


# cat=0
# dog=0
# for sample in training_data:
#     if sample[1] == 0:
#         cat =cat+1
#     else:
#         dog=dog+1

# print(f"cat:{cat} dog:{dog}")


#shuffle data

import random

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])


x=[]
y=[]
#creating traning dataset
for features,label in training_data:
    x.append(features)
    y.append(label)


x=np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,1)


import pickle

pickle_out = open("x.pickle","wb")
pickle.dump(x,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()


