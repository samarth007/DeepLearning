import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense,Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.activations import relu,sigmoid
from sklearn.metrics import confusion_matrix
from time import time
t=time()
data=pd.read_csv('E:\pythonProject\Churn\Churn_Modelling.csv')
df=data.drop(['RowNumber','CustomerId','Surname'],axis=1)


for i in df.columns:
    if (df[i].dtypes!='object'):
        lower=df[i].mean() - 3 * df[i].std()
        upper=df[i].mean() + 3 * df[i].std()
        df=df[(df[i] > lower) & (df[i] < upper)]

geo=pd.get_dummies(df[['Geography','Gender']],drop_first=True)
df=pd.concat([df.drop(['Geography','Gender'],axis=1),geo],axis=1)
X=df.drop('Exited',axis=1)
Y=df['Exited']

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=Sequential()
model.add(Dense(units=10,kernel_initializer='he_uniform',activation=relu,input_shape=(X.shape[1],),name='Input'))
model.add(Dense(units=8,kernel_initializer='he_uniform',activation=relu,name='First_layer'))
model.add(Dense(units=5,kernel_initializer='he_uniform',activation=relu,name='Second_layer'))
model.add(Dense(units=1,kernel_initializer='glorot_uniform',activation=sigmoid,name='Output'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,validation_split=0.3,epochs=15,batch_size=50)
y_pred=model.predict(x_test)
y_pred=np.where(y_pred>0.5,1,0)
cm=confusion_matrix(y_pred,y_test)
print(cm)
print(t-time())