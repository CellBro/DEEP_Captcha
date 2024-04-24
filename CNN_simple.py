import copy

import cv2
import pandas
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array

path = 'data/train'


# Converting images to appropriate samples
def DIP(img1):
    # 阈值化
    thresh_img1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)

    # 闭操作（先膨胀后腐蚀）
    close_img1 = cv2.morphologyEx(thresh_img1, cv2.MORPH_CLOSE, np.ones((5, 2), np.uint8))

    # 膨胀
    dilate_img1 = cv2.dilate(close_img1, np.ones((2, 2), np.uint8), iterations=1)

    # 平滑
    gauss_img1 = cv2.GaussianBlur(dilate_img1, (1, 1), 0)
    return gauss_img1


X = []
y = []

for image in os.listdir(path):

    if not image.endswith(".png"):
        continue

    img = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
    img = DIP(img)
    image_list = [img[10:70, 10:50], img[10:70, 50:90], img[10:70, 90:130], img[10:70, 130:170]]

    for i in range(4):
        X.append(img_to_array(Image.fromarray(image_list[i])))
        y.append(image[i])
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)
X /= 255.0

# Initial Analysis and Data Wrangling

# plt.figure(figsize = (20,8))
# for i in range(5) :
#     plt.subplot(1,5,i+1)
#     plt.imshow(X[i], 'gray')
#     plt.title('Label is ' + str(y[i]))
# plt.plot()
# plt.waitforbuttonpress()

# temp = set(y)
# for t in temp :
#     print('Occurance count of ' + t + ' : ' + str(len(y[y == t])))
# temp_df = pandas.DataFrame({'labels': [t for t in temp], 'Count': [len(y[y == t]) for t in temp]})
# plt.figure(figsize = (20,8))
# seaborn.barplot(x = 'labels', y = 'Count', data = temp_df, palette = 'Blues_d')
# plt.title('Label distribution in CAPTCHAS', fontsize = 20)
# plt.waitforbuttonpress()
## 分析得不需要过采样


# One hot encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

lbenc = LabelEncoder()

y_combine = lbenc.fit_transform(y)

one_hot_enc = OneHotEncoder(sparse_output=False)
temp_y = copy.deepcopy(y_combine)
y_one_hot = one_hot_enc.fit_transform(np.reshape(temp_y, (len(y_combine), 1)))

# print(y_one_hot)

# print(y)
# print('letter n : ' + str(y[1]))
# print('label : ' + str(y_combine[1]))
# print('Count : ' + str(len(y_combine[y_combine == y_combine[1]])))
info = {y_combine[i]: y[i] for i in range(len(y))}
# print(X.shape)
# print(y_one_hot.shape)  # one hot encoded form

# Train Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=1)

# y_temp = np.argmax(y_test, axis=1)
# temp = set(y_temp)
# temp_df = pandas.DataFrame({'labels': [info[t] for t in temp], 'Count': [len(y_temp[y_temp == t]) for t in temp]})
# plt.figure(figsize=(20, 8))
# seaborn.barplot(x='labels', y='Count', data=temp_df, palette='Blues_d')
# plt.title('Label distribution in test set', fontsize=20)
# plt.waitforbuttonpress


# Model Creation

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization

from keras.layers import Dropout
from keras.layers import Input

print(X_train.shape)
print(y_train.shape)


#
# print(X_test.shape)
# print(y_test.shape)


def conv_layer(filterx):
    model = Sequential()

    model.add(Conv2D(filterx, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    return model


def dens_layer(hiddenx):
    model = Sequential()

    model.add(Dense(hiddenx, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

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


#
# plt.figure(figsize = (60,40))
#
# hi = 7200
# lo = 2400
#
# for i in range(25) :
#     plt.subplot(5,5,i+1)
#     x = np.random.randint(lo, hi)
#     plt.imshow(X_train[x], 'gray')
#     plt.title('Label is ' + str(info[np.argmax(y_train[x])]))
# plt.show()
# plt.waitforbuttonpress()

# data generator
traingen = ImageDataGenerator(rotation_range=5, width_shift_range=[-2, 2])
traingen.fit(X_train)
train_set = traingen.flow(X_train, y_train)

trainX, trainy = train_set.next()

# training

model = cnn(128, 32, 16, 32, 32)
model.summary()
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

checkp = ModelCheckpoint('log/result_model_62.h5', monitor='val_loss', verbose=1, save_best_only=True)
reduce = ReduceLROnPlateau(monitor='val_loss', patience=20, verbose=1)
print(X_train.shape)
print(y_train.shape)

history = model.fit(traingen.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=200,
                    steps_per_epoch=len(X_train) / 32, callbacks=[checkp])
#
plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.legend(['train loss', 'val loss'])
plt.title('Loss function wrt epochs')

plt.subplot(2, 1, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train acc', 'val acc'])
plt.title('Model accuracy wrt Epoch')
plt.pause(1)

# 模型效果展示
from keras.models import load_model

model = load_model('log/result_model_62.h5')
pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
yres = np.argmax(y_test, axis=1)
from sklearn.metrics import accuracy_score, classification_report

target_name = []
for i in sorted(info):
    target_name.append(info[i])
print('Accuracy : ' + str(accuracy_score(yres, pred)))
print(classification_report(yres, pred, target_names=target_name))

#
# def get_demo(img_path):
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#
#     plt.imshow(img, 'gray')
#     plt.axis('off')
#     plt.show()
#
#     img = DIP(img)
#
#     image_list = [img[10:50, 30:50], img[10:50, 50:70], img[10:50, 70:90], img[10:50, 90:110], img[10:50, 110:130]]
#
#     plt.imshow(img, 'gray')
#     plt.axis('off')
#     plt.show()
#     Xdemo = []
#     for i in range(5):
#         Xdemo.append(img_to_array(Image.fromarray(image_list[i])))
#
#     Xdemo = np.array(Xdemo)
#     Xdemo /= 255.0
#
#     ydemo = model.predict(Xdemo)
#     ydemo = np.argmax(ydemo, axis=1)
#
#     for res in ydemo:
#         print(info[res])
#     print(img_path[-9:])
#
# get_demo('../input/captcha-version-2-images/samples/2g783.png')
# get_demo('../input/captcha-version-2-images/samples/2fxgd.png')
# get_demo('../input/captcha-version-2-images/samples/88bgx.png')
# get_demo('../input/captcha-version-2-images/samples/4yc85.png')
