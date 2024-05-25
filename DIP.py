import cv2
import pandas
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

path1 = "data/train/2lpF_209.png"
path2 = 'data/train/2Zku_875.png'


def pp(img):
    plt.imshow(img)
    plt.pause(0.1)


def plot_(img1, img2):
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(img1, 'gray')

    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2, 'gray')

    plt.axis('off')
    plt.pause(0.1)


img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

# 阈值化
thresh_img1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)
thresh_img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)

# # 闭操作（先膨胀后腐蚀）
# close_img1 = cv2.morphologyEx(thresh_img1, cv2.MORPH_CLOSE, np.ones((5, 2), np.uint8), iterations=1)
# close_img2 = cv2.morphologyEx(thresh_img2, cv2.MORPH_CLOSE, np.ones((5, 2), np.uint8), iterations=1)
#
# #
# # 开操作（先膨胀后腐蚀）
# open_img1 = cv2.morphologyEx(close_img1, cv2.MORPH_OPEN, np.ones((5, 2), np.uint8), iterations=1)
# open_img2 = cv2.morphologyEx(close_img2, cv2.MORPH_OPEN, np.ones((5, 2), np.uint8), iterations=1)
#
#
# median_img1 = cv2.medianBlur(thresh_img1, ksize=5)
# median_img2 = cv2.medianBlur(thresh_img2, ksize=5)

# # 膨胀
# dilate_img1 = cv2.dilate(median_img1, np.ones((2, 2), np.uint8), iterations=1)
# dilate_img2 = cv2.dilate(median_img2, np.ones((2, 2), np.uint8), iterations=1)

# # 平滑
# gauss_img1 = cv2.GaussianBlur(dilate_img1, (1, 1), 0)
# gauss_img2 = cv2.GaussianBlur(dilate_img2, (1, 1), 0)


def get_images_cor(img):
    pp(img)
    '''
    垂直投影为从上往下投射，统计每一列的黑色像素总数
    '''
    rows, cols = img.shape
    ver_list = [0] * cols
    for j in range(cols - 25):
        for i in range(10, rows - 10):
            if img.item(i, j) == 0:
                ver_list[j] = ver_list[j] + 1
    '''
    对ver_list中的元素进行筛选，可以去除一些噪点
    '''
    ver_arr = np.array(ver_list)
    ver_arr[np.where(ver_arr < 8)] = 0
    ver_list = ver_arr.tolist()
    # print(ver_list)
    # print([i for i in range(0,200)])
    '''
    分割字符
    '''
    w_list = {}
    last = 0
    begin = 0
    for index, i in enumerate(ver_list):
        if i * 3 < last and last:
            last = i
            back = index
            w = sum(ver_list[begin:back])
            w_list[w] = (begin, back)
            begin = back
        else:
            last = i
    # '''
    # 得到横坐标
    # '''
    x_list = []

    # 按子列表的长度排序
    sorted_lists = sorted(w_list.items(), key=lambda d: d[0], reverse=True)

    # 选取前四个子列表
    top_four = sorted_lists[:4]
    # print(top_four)
    for weight, (a, b) in top_four:
        # print(weight, a, b)
        cent = (a + b) // 2
        x_list.append(cent)
    return list(map(int, sorted(x_list)))


img = thresh_img1
X_list = get_images_cor(img)
print(X_list)
img_list = [img[10:70, X_list[0] - 20:X_list[0] + 20], img[10:70, X_list[1] - 20:X_list[1] + 20],
            img[10:70, X_list[2] - 20:X_list[2] + 20], img[10:70, X_list[3] - 20:X_list[3] + 20]]

for i in img_list:
    pp(i)
exit(1)

# cv2.rectangle(gauss_img1, (10,10), (50,70), 0, 1)
# cv2.rectangle(gauss_img1, (50,10), (90,70), 0, 1)
# cv2.rectangle(gauss_img1, (90,10), (130,70), 0, 1)
# cv2.rectangle(gauss_img1, (130,10), (170,70),0, 1)
#
#
# cv2.rectangle(gauss_img2, (10,10), (55,70), 0, 1)
# cv2.rectangle(gauss_img2, (55,10), (100,70), 0, 1)
# cv2.rectangle(gauss_img2, (100,10), (145,70), 0, 1)
# cv2.rectangle(gauss_img2, (145,10),(190,70),0, 1)
#
# plot_(gauss_img1, gauss_img2)
