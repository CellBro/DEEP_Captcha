import cv2
import numpy as np
from PIL import Image
from keras import Sequential, Input
from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Dense, Flatten
from keras.utils import img_to_array
from matplotlib import pyplot as plt

l_to_n={
    '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
    'a':10,'b':11,'c':12,'d':13,'e':14,'f':15,'g':16,'h':17,'i':18,'j':19,'k':20,'l':21,'m':22,'n':23,'o':24,'p':25,'q':26,'r':27,'s':28,'t':29,'u':30,'v':31,'w':32,'x':33,'y':34,'z':35,
    'A':36,'B':37,'C':38,'D':39,'E':40,'F':41,'G':42,'H':43,'I':44,'J':45,'K':46,'L':47,'M':48,'N':49,'O':50,'P':51,'Q':52,'R':53,'S':54,'T':55,'U':56,'V':57,'W':58,'X':59,'Y':60,'Z':61,

}



def conv_layer(filterx):
    model = Sequential()

    model.add(Conv2D(filterx, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    return model

def get_images_cor(img):

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
        x_list.append(cent+5)
    return list(map(int, sorted(x_list)))
def dens_layer(hiddenx):
    model = Sequential()

    model.add(Dense(hiddenx, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    return model


def cnn(filter1, filter2, filter3, hidden1, hidden2):
    model = Sequential()
    model.add(Input((60, 40, 1,)))

    model.add(conv_layer(filter1))
    model.add(conv_layer(filter2))
    model.add(conv_layer(filter3))

    model.add(Flatten())
    model.add(dens_layer(hidden1))
    model.add(dens_layer(hidden2))

    model.add(Dense(62, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def DIP(img1):
    # 阈值化
    thresh_img1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)

    # # 闭操作（先膨胀后腐蚀）
    # close_img1 = cv2.morphologyEx(thresh_img1, cv2.MORPH_CLOSE, np.ones((5, 2), np.uint8))
    #
    # open_img1 = cv2.morphologyEx(close_img1, cv2.MORPH_OPEN, np.ones((3, 2), np.uint8), iterations=1)

    # median_img1 = cv2.medianBlur(thresh_img1, ksize=5)
    # # 膨胀
    # dilate_img1 = cv2.dilate(median_img1, np.ones((2, 2), np.uint8), iterations=1)

    # # 平滑
    # gauss_img1 = cv2.GaussianBlur(dilate_img1, (1, 1), 0)
    return thresh_img1
def get_demo(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    plt.imshow(img, 'gray')
    plt.axis('off')
    plt.show()

    img = DIP(img)
    try:
        X_list = get_images_cor(img)
        if len(X_list) != 4:
            print(f"{img_path} was discarded")
        image_list = []
        image_list = np.zeros((4, 60, 40))
        for i in range(4):
            temp = img[10:70, max(0, X_list[i] - 20):min(200, X_list[i] + 20)]
            if np.shape(temp)[1] != 40:
                col = 40 - np.shape(temp)[1]
                temp = np.pad(temp, ((0, 0), (0, col)), mode='constant')
            image_list[i] = temp

        # 均匀分割
        # image_list = [img[10:70, 10:50], img[10:70, 50:90], img[10:70, 90:130], img[10:70, 130:170]]
        # plt.imshow(img)
        # print(image)
        # plt.pause(0.1)

    except:
        print("Error!\n")

    plt.imshow(img, 'gray')
    plt.axis('off')
    plt.show()
    Xdemo = []
    for i in range(4):
        Xdemo.append(img_to_array(Image.fromarray(image_list[i])))

    Xdemo = np.array(Xdemo)
    Xdemo /= 255.0
    model = cnn(128, 32, 16, 32, 32)
    ydemo = model.predict(Xdemo)
    ydemo = np.argmax(ydemo, axis=1)

    for res in ydemo:
        print(res)
    print(img_path[-9:])


get_demo("data/train/0jJw_1635.png")