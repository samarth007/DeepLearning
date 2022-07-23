import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense,Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.activations import relu,sigmoid
from sklearn.metrics import confusion_matrix
from scipy import stats

data=pd.read_csv('D:\DeepLearning\Churn\Churn_Modelling.csv')
geo=pd.get_dummies(data[['Geography','Gender']],drop_first=True)
df=pd.concat([data,geo],axis=1)
df=data.drop(['RowNumber','CustomerId','Surname','Geography','Gender'],axis=1)

df=df[(np.abs(stats.zscore(df))<3).all(axis=1)] #removing outliers using zscore

# Detecting outlier in each feature
# def detectOutliers(d,feature):
#   q1=d[feature].quantile(0.25)
#   q3=d[feature].quantile(0.75)
#   IQR=q3-q1
#   lower_bound=q1-(1.5*IQR)
#   upper_bound=q3+(1.5*IQR)
#   # print(feature,lower_bound,upper_bound)
#   lst=d.index[(d[feature]<lower_bound) | (d[feature]>upper_bound)]
#   # print(feature,lst)
#   return lst

# outlier_index_list=[]
# for i in df:
#   outlier_index_list.extend(detectOutliers(df,i))

# #To remove outliers from row
# def remove_outlier(d,outlierIndex):
#   sortedList=sorted(set(outlierIndex))
#   d=d.drop(sortedList)
#   return d

# df=remove_outlier(df,outlier_index_list)


X=df.drop('Exited',axis=1)
Y=df['Exited']

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=Sequential()
model.add(Dense(units=10,kernel_initializer='he_uniform',activation=relu,input_shape=(X.shape[1],),name='Input'))
model.add(Dense(units=5,kernel_initializer='he_uniform',activation=relu,name='First_layer'))
model.add(Dense(units=3,kernel_initializer='he_uniform',activation=relu,name='Second_layer'))
model.add(Dense(units=1,kernel_initializer='glorot_uniform',activation=sigmoid,name='Output'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=15,batch_size=20)
y_pred=model.predict(x_test)
y_pred=np.where(y_pred>0.5,1,0)
cm=confusion_matrix(y_pred,y_test)
print(cm)
