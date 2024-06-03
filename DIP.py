import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

data_path = "data/train"
save_path = "torch_data/train"
l_to_n={
    '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
    'a':10,'b':11,'c':12,'d':13,'e':14,'f':15,'g':16,'h':17,'i':18,'j':19,'k':20,'l':21,'m':22,'n':23,'o':24,'p':25,'q':26,'r':27,'s':28,'t':29,'u':30,'v':31,'w':32,'x':33,'y':34,'z':35,
    'A':36,'B':37,'C':38,'D':39,'E':40,'F':41,'G':42,'H':43,'I':44,'J':45,'K':46,'L':47,'M':48,'N':49,'O':50,'P':51,'Q':52,'R':53,'S':54,'T':55,'U':56,'V':57,'W':58,'X':59,'Y':60,'Z':61,
}
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


def get_images_cor(img):
    '''
    垂直投影为从上往下投射，统计每一列的黑色像素总数
    '''
    rows, cols = img.shape
    ver_list = [0] * cols
    for j in range(15,cols - 25):
        for i in range(10, rows - 10):
            if img.item(i, j) == 0:
                ver_list[j] = ver_list[j] + 1
    '''
    对ver_list中的元素进行筛选，可以去除一些噪点
    '''
    ver_arr = np.array(ver_list)
    ver_arr[np.where(ver_arr < 9)] = 0
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
        x_list.append(cent+2)
    return list(map(int, sorted(x_list)))


cnt = 0
for image in os.listdir(data_path):
    cnt = cnt + 1
    if cnt % 100 == 0:
        print(f"Working on data {cnt}")
    if not image.endswith(".png"):
        continue
    img_ori = cv2.imread(os.path.join(data_path, image), cv2.IMREAD_GRAYSCALE)
    img = cv2.adaptiveThreshold(img_ori, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 141, 0)
    median_img1 = cv2.medianBlur(img, ksize=5)
    gauss_img1 = cv2.GaussianBlur(median_img1, (1, 1), 0)

    # try:
    X_list = get_images_cor(gauss_img1)
    if len(X_list) != 4:
        print(f"{image} was discarded")
        continue
    image_list = np.zeros((4, 60, 40))
    for i in range(4):
        temp = img[10:70, max(0, X_list[i] - 20):min(200, X_list[i] + 20)]
        if np.shape(temp)[1] != 40:
            col = 40 - np.shape(temp)[1]
            temp = np.pad(temp, ((0, 0), (0, col)), mode='constant',constant_values=(255,255))
        image_list[i] = temp

    filename = str(image).split('/')[-1].split('_')[0]
    for idx,img in enumerate(image_list):
        dir_path=os.path.join(save_path, f"{l_to_n[filename[idx]]}")
        if not os.path.exists(dir_path):
            os.mkdir(f"{dir_path}")
        cv2.imwrite(os.path.join(dir_path, f"{filename}_{filename[idx]}_.png"), img)

    # except Exception as e:
    #     print(f"Warning!{e}")
print("Image processing finished")
