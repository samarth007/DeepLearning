from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv(r'D:\DeepLearning\Boston\book.csv')
X=data.drop('labels',axis=1)
Y=data.labels
print(X.shape[1])
model=Sequential()
model.add(Dense(units=10,activation='relu',kernel_initializer='he_uniform',input_shape=(X.shape[1],)))
model.add(Dense(units=5,activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(units=1,activation='sigmoid',kernel_initializer='glorot_uniform'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X,Y,epochs=10)
print(model.summary())