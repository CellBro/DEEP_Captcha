import cv2
import pandas
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

path1 = "data/train/0Ffs_161.png"
path2 = 'data/train/0iHV_1159.png'


def plot_(img1, img2):
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(img1, 'gray')

    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2, 'gray')

    plt.axis('off')
    plt.waitforbuttonpress()


img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

# 阈值化
thresh_img1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)
thresh_img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)
# 闭操作（先膨胀后腐蚀）
close_img1 = cv2.morphologyEx(thresh_img1, cv2.MORPH_CLOSE, np.ones((5, 2), np.uint8))
close_img2 = cv2.morphologyEx(thresh_img2, cv2.MORPH_CLOSE, np.ones((5, 2), np.uint8))

# 膨胀
dilate_img1 = cv2.dilate(close_img1, np.ones((2, 2), np.uint8), iterations=1)
dilate_img2 = cv2.dilate(close_img2, np.ones((2, 2), np.uint8), iterations=1)

# 平滑
gauss_img1 = cv2.GaussianBlur(dilate_img1, (1,1), 0)
gauss_img2 = cv2.GaussianBlur(dilate_img2, (1,1), 0)

cv2.rectangle(gauss_img1, (10,10), (50,70), 0, 1)
cv2.rectangle(gauss_img1, (50,10), (90,70), 0, 1)
cv2.rectangle(gauss_img1, (90,10), (130,70), 0, 1)
cv2.rectangle(gauss_img1, (130,10), (170,70),0, 1)


cv2.rectangle(gauss_img2, (10,10), (55,70), 0, 1)
cv2.rectangle(gauss_img2, (55,10), (100,70), 0, 1)
cv2.rectangle(gauss_img2, (100,10), (145,70), 0, 1)
cv2.rectangle(gauss_img2, (145,10),(190,70),0, 1)

plot_(gauss_img1, gauss_img2)
