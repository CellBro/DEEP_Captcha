import torch
import cv2
path="torch_data/train/0/6R02_0_.png"
img=cv2.imread(path)
channels=cv2.split(img)
cv2.imshow("red",channels[2])
cv2.waitKey(0)