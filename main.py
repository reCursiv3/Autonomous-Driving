# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 20:49:46 2021

@author: Aadesh
"""

import sklearn
import pickle
import os
import cv2
import numpy as np
import skimage
from skimage import data,color,exposure
import matplotlib.pyplot as plt
from collections import deque
from scipy.ndimage.measurements import label
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D, Flatten, Input, Conv2D, MaxPooling2D,Lambda
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage import io
from IPython.display import HTML
import glob

directory="D:/vehicle_detection/dataset_gti"
main_category=["non-vehicles","vehicles"]
sub_categories=["Far","Left","MiddleClose","Right"]

cars=glob.glob("D:/vehicle_detection/dataset_gti/vehicles/*/*.png")
non_cars=glob.glob("D:/vehicle_detection/dataset_gti/non-vehicles/*/*.png")

X=[]
for file in cars:
    X.append(io.imread(file))
for file in non_cars:
    X.append(io.imread(file))
X=np.array(X)
Y=np.concatenate([np.ones(len(cars)),np.zeros(len(non_cars))])
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=60)

def showImages():
    fig=plt.figure(figsize=(12,6))
    for i in range(0,40):
        number=np.random.randint(0,len(X_train))
        axis=fig.add_subplot(4,10,i+1)
        axis.set_xlabel(Y_train[number])
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.imshow(X_train[number])
    plt.show()
showImages()

def vehicle_network(inputShape=(64,64,3)):
    model=Sequential()
    model.add(Lambda(lambda x: x/127.5-1.,input_shape=inputShape,output_shape=inputShape))
    model.add(Conv2D(filters=16,kernel_size=(3,3),activation='relu',name='Convolution0',input_shape=inputShape,padding='same'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',name='Convolution1',input_shape=inputShape,padding='same'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',name='Convolution2',input_shape=inputShape,padding='same'))
    model.add(MaxPooling2D(pool_size=(8,8)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=1,kernel_size=(8,8),name='Final_Convolution',activation='sigmoid'))
    return model

model=vehicle_network()
model.summary()
model.add(Flatten())

model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
finale=model.fit(X_train,Y_train,batch_size=32,nb_epoch=20,verbose=2,validation_data=(X_test,Y_test))
model.save_weights('./model.h5')

rand=np.random.randint(X_test.shape[0])
plt.imshow(X_test[rand])
sample=np.reshape(X_test[rand],(1,64,64,3))
prediction=model.predict(sample,batch_size=64,verbose=0)
prediction=prediction[0][0]
if prediction>=0.5:
    print("Network predicted: CAR ; Probability: "+str(prediction))
else:
    print("Network predicted: NO CAR ; Probability: "+str(prediction))
truth=Y_test[rand]
if truth==1:
    print("Ground truth: CAR")
else:
    print("Ground truth: NO CAR")

img=io.imread('./testing/test_image.png')
fig=plt.figure(figsize=(12,20))
plt.imshow(img)

def make_box(img,boxes,color=(0,0,255),thick=6):
    draw_img=np.copy(img)
    for box in boxes:
        cv2.rectangle(draw_img,box[0],box[1],color,thick)
    return draw_img

def detection(img):
    cropped=img[400:1000,0:1280]
    heat=heatmodel.predict(cropped.reshape(1,cropped.shape[0],cropped.shape[1],cropped.shape[2]))
    xx,yy=np.meshgrid(np.arange(heat.shape[2]),np.arange(heat.shape[1]))
    x=(xx[heat[0,:,:,0]>0.9999999])
    y=(yy[heat[0,:,:,0]>0.9999999])
    hot_windows=[]
    for i,j in zip(x,y):
        hot_windows.append(((i*8,400+j*8),(i*8+64,400+j*8+64)))
    return hot_windows

heatmodel=vehicle_network((600,1280,3))
heatmodel.load_weights('./model.h5')
hot_windows= detection(img)
window_img=make_box(img,hot_windows,(0,255,0),6)
fig=plt.figure(figsize=(12,20))
plt.imshow(window_img)
