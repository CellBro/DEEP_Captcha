import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from models import *
l_to_n={
    '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
    'a':10,'b':11,'c':12,'d':13,'e':14,'f':15,'g':16,'h':17,'i':18,'j':19,'k':20,'l':21,'m':22,'n':23,'o':24,'p':25,'q':26,'r':27,'s':28,'t':29,'u':30,'v':31,'w':32,'x':33,'y':34,'z':35,
    'A':36,'B':37,'C':38,'D':39,'E':40,'F':41,'G':42,'H':43,'I':44,'J':45,'K':46,'L':47,'M':48,'N':49,'O':50,'P':51,'Q':52,'R':53,'S':54,'T':55,'U':56,'V':57,'W':58,'X':59,'Y':60,'Z':61,
}
n_to_l = {v : k for k, v in l_to_n.items()}
classes = {'0': 0, '1': 1, '2': 10, '3': 11, '4': 12, '5': 13, '6': 14, '7': 15, '8': 16, '9': 17, '10': 18, '11': 19, '12': 2, '13': 20, '14': 21, '15': 22, '16': 23, '17': 24, '18': 25, '19': 26, '20': 27, '21': 28, '22': 29, '23': 3, '24': 30, '25': 31, '26': 32, '27': 33, '28': 34, '29': 35, '30': 36, '31': 37, '32': 38, '33': 39, '34': 4, '35': 40, '36': 41, '37': 42, '38': 43, '39': 44, '40': 45, '41': 46, '42': 47, '43': 48, '44': 49, '45': 5, '46': 50, '47': 51, '48': 52, '49': 53, '50': 54, '51': 55, '52': 56, '53': 57, '54': 58, '55': 59, '56': 6, '57': 60, '58': 61, '59': 7, '60': 8, '61': 9}

data_path = "./test"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
modelname="VGG_lr_0.1_ckpt.pth"



print('==> Building model..')
net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net.to(device)
checkpoint = torch.load(f'./checkpoint/{modelname}',map_location=device)

from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in checkpoint["net"].items():
    name = k.replace('module.', '')  # remove `module.`
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
best_acc = checkpoint['acc']
print(best_acc)


transform = transforms.Compose([
transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
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


cnt = 0.0
r_cnt=0.0

id_key={}

for image in os.listdir(data_path):
    cnt = cnt + 1
    if cnt % 100 == 0:
        print(f"Working on data {cnt}")
    if not image.endswith(".png"):
        continue

    id = int(image.split('.')[0])



    img_ori = cv2.imread(os.path.join(data_path, image), cv2.IMREAD_GRAYSCALE)
    img = cv2.adaptiveThreshold(img_ori, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 141, 0)
    # median_img1 = cv2.medianBlur(img, ksize=5)
    # gauss_img1 = cv2.GaussianBlur(median_img1, (1, 1), 0)
    # try:
    X_list = get_images_cor(img)

    if len(X_list) != 4:
        print(f"{image} was discarded")
        key = "!!!!"
        id_key[id]=key
        continue


    image_list = np.zeros((4, 60, 40))
    for i in range(4):
        temp = img[10:70, max(0, X_list[i] - 20):min(200, X_list[i] + 20)]
        if np.shape(temp)[1] != 40:
            col = 40 - np.shape(temp)[1]
            temp = np.pad(temp, ((0, 0), (0, col)), mode='constant',constant_values=(255,255))
        image_list[i] = temp

    pre=[]
    # label=str(image.split('_')[0])
    for img in image_list:
        image_tensor = transform(Image.fromarray(img))
        image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
        image_tensor = image_tensor.to(device)
        net.eval()
        with torch.no_grad():
            output=net(image_tensor)
        predicted_classes = torch.argmax(output, dim=1)
        predicted_class_number = predicted_classes.item()

        # 将数字转化为原始的字符或字母
        predicted_label = n_to_l[classes[str(predicted_class_number)]]
        pre.append(predicted_label)
    pre="".join(pre)
    id_key[id]=pre

    # print(pre)
    # plt.imshow(img_ori)
    # plt.pause(1)
    # print(pre)
    # print(label)
    # if pre == label:
    #     r_cnt=r_cnt+1


with open("22920212204263_new.txt", 'w') as f:
    for id,key in id_key.items():
        f.write(f"{id}\t{key}\n")
        f.flush()




