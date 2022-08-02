import numpy as np
import cv2
import os
import tensorflow.python.keras.models
from sklearn.metrics import confusion_matrix,accuracy_score
import pandas as pd

new_model = tensorflow.keras.models.load_model('E:\pythonProject\Classification\my_model.h5')
path = r"E:\pythonProject\Classification\validateSet"

X = []
y = []
convert = lambda ss: int(ss == 'dog')


def create_test_data(path):  # to create X and Y variable and converting
    for p in os.listdir(path):
        category = p.split(".")[0]
        category = convert(category)
        img_array = cv2.imread(os.path.join(path, p), cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X.append(new_img_array)
        y.append(category)


create_test_data(path)

X = np.array(X).reshape(-1, 80, 80, 1)
X = X / 255
y_pred=new_model.predict(X)
y_pred=np.where(y_pred>0.5,1,0)
cm=confusion_matrix(y_pred,y)
acc=accuracy_score(y,y_pred)
print(cm)

df_pred=pd.DataFrame(y_pred)
df_actual=pd.DataFrame(y)
df=pd.concat([df_pred,df_actual],axis=1)
