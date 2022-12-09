import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2

data_dir = "dataset/training_set"
categories = ["cats", "dogs"]
img_size = 50
training_data=[]
for category in categories:
    path = os.path.join(data_dir, category) #path to cats or dogs dir
    class_num = categories.index(category) #labeling the classes 0 or 1
    for img in os.listdir(path):
        try:
            img_array= cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_size, img_size))
            training_data.append([new_array,class_num])
        except Exception as e:
            pass
    #     plt.imshow(img_array,cmap="gray")
    #     plt.show()
    #     break
    # break

#shaping
print(len(training_data))

# new_array = cv2.resize(img_array, (img_size, img_size))

# plt.imshow(new_array,cmap="gray")
# plt.show()


import random
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])


x=[]
y=[]

for f,l in training_data:
    x.append(f)
    y.append(l)

X = np.array(x).reshape(-1, img_size, img_size, 1)


