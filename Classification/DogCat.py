import os
import cv2
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from tensorflow.python.keras.activations import relu,sigmoid
import pandas as pd
import numpy as np

path='E:\pythonProject\Classification\Trains'
filename=os.listdir(path)

# label=[]
# for f in filename:
#     if f.split('.')[0]=='cat':
#         label.append('cat')
#     else:
#         label.append('dog')

# df=pd.DataFrame({'name':filename,'label':label})
# X=df['name']
# Y=df['label']

#
# for p in os.listdir(path):  #for Visualisation of each image
#     category = p.split(".")[0]
#     img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
#     new_img_array = cv2.resize(img_array, dsize=(80, 80))
#     plt.imshow(new_img_array,cmap='gray')
#     break
# plt.show()

X = []
y = []
convert = lambda ss : int(ss == 'dog')
def create_test_data(path):  #to create X and Y variable and converting
    for p in os.listdir(path):
        category = p.split(".")[0]
        category = convert(category)
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X.append(new_img_array)
        y.append(category)

create_test_data(path)

X=np.array(X).reshape(-1,80,80,1)
y=np.array(y)

X=X/255  #normalize

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=23)

model=Sequential()
model.add(Conv2D(64,(3,3),activation=relu,input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation=relu))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64,activation=relu))

model.add(Dense(1,activation=sigmoid))
model.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')

model.fit(x_train,y_train,validation_split=0.3,batch_size=10,epochs=10)
y_pred=model.predict(x_test)
y_pred=np.where(y_pred>0.5,1,0)


# model.save('my_model.h5')
