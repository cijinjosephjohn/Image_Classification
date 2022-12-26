IMG_SIZE = 50

new_arr = cv2.resize(img_arr,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_arr,cmap="gray")