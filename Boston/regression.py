import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.activations import relu,linear
from tensorflow.python.keras.metrics import MeanSquaredError,MeanAbsoluteError
from scipy import stats


raw_data=load_boston()
Initial_X=pd.DataFrame(data=raw_data.data,columns=raw_data.feature_names)
Initial_X=Initial_X.drop('CHAS',axis=1) #Most of the values are zero so dropping the feature
Initial_Y=pd.DataFrame(data=raw_data.target,columns=['price'])
data=pd.concat([Initial_X,Initial_Y],axis=1)

print(data.shape)

df=data[(np.abs(stats.zscore(data))<3).all(axis=1)]  #removing outlier

# To visvualize outlier in each feature
def plot_boxplot(df,feature):
   sb.boxplot(df[feature])
   plt.grid(False)
   plt.show()

for i in data:
   plot_boxplot(data,i)

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
# for i in data:
#   outlier_index_list.extend(detectOutliers(data,i))

#To remove outliers from each feature
# def remove_outlier(d,outlierIndex):
#   sortedList=sorted(set(outlierIndex))
#   d=d.drop(sortedList)
#   return d

# df=remove_outlier(data,outlier_index_list)
print(df.shape)

X=df.drop('price',axis=1)
Y=df['price']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.15,random_state=42)
scaler=StandardScaler()
x_train_transformed=scaler.fit_transform(x_train)
x_test_transformed=scaler.transform(x_test)

cb=EarlyStopping(monitor='loss',patience=3,mode='min')
model=Sequential()
model.add(Dense(units=10,kernel_initializer='normal',activation=relu,input_shape=(X.shape[1],)))
model.add(Dense(units=8,kernel_initializer='normal',activation=relu))
model.add(Dense(units=5,kernel_initializer='normal',activation=relu))
model.add(Dense(units=1,kernel_initializer='normal',activation=linear))
model.compile(optimizer='adam',loss='mae',metrics=[MeanAbsoluteError()])
model.fit(x_train_transformed,y_train,batch_size=1,epochs=200,callbacks=[cb]) #Stochastic gradient descent
y_pred=model.predict(x_test_transformed)

# actual=df.loc[x_test.index]
# predicted=pd.DataFrame(y_pred,index=x_test.index)
# pred_dataframe=pd.concat([actual,predicted],axis=1)
# pred_dataframe.to_csv('pred_df.csv',index=False)