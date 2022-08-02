import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score,recall_score,plot_confusion_matrix,classification_report,\
    confusion_matrix,ConfusionMatrixDisplay
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.activations import sigmoid,relu
from tensorflow.python.keras.metrics import BinaryAccuracy,FalseNegatives,Precision,Recall,FalsePositives
from tensorflow.python.keras.losses import binary_crossentropy,BinaryCrossentropy
from scipy import stats


df=pd.read_csv('D:\DeepLearning\Diabetic\diabetes.csv')
print(df.shape)

# #Checking for balance in dataset
# outcome1_count=len(data.loc[data['Outcome']==1])
# outcome0_count=len(data.loc[data['Outcome']==0])
# print(outcome0_count)
# print(outcome1_count)

df=df[(np.abs(stats.zscore(df))<3).all(axis=1)]  #removing outliers
print(df.shape)

# def detect_outlier(data):
#     for i in data:
#       upperlimit=data[i].mean() + 3 * data[i].std()
#       lowerlimit=data[i].mean() - 3 * data[i].std()
#       data=data[(data[i] > lowerlimit) & (data[i] < upperlimit)]
#     return data

# df=detect_outlier(df)
# print(df.shape)

#Independent and Dependent
X=df.drop('Outcome',axis=1)
Y=df['Outcome']

# corr=df.corr()
# sb.heatmap(data=corr,annot=True,cmap='RdYlGn')
# cor_target=abs(corr['Outcome'])
# relevant_feature=corr[cor_target>0.5]
# plt.show()

# for i in X:
#     print('Missing values in '+i,df[i].isin([0]).sum())
#     col=df[i].isin([0]).sum()/len(df[i]) * 100
#     print('Percentage missing in '+i,round(col,2))



x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=10,test_size=0.2)
scaler=StandardScaler()
x_train_transformed=scaler.fit_transform(x_train)
x_test_transformed=scaler.transform(x_test)

def create_network():
    model=Sequential()
    model.add(Dense(units=10,kernel_initializer='he_uniform',activation=relu,input_shape=(X.shape[1],)))
    model.add(Dense(units=8,kernel_initializer='he_uniform',activation=relu))
    model.add(Dense(units=5,kernel_initializer='he_uniform',activation=relu))
    model.add(Dense(units=1,kernel_initializer='glorot_uniform',activation=sigmoid))
    model.compile(optimizer='adam',loss=BinaryCrossentropy(),metrics=['accuracy',Recall()])
    return model
kc=KerasClassifier(build_fn=create_network,epochs=10,batch_size=1,)

# model=create_network()
# model.compile(optimizer='adam',loss=BinaryCrossentropy(),metrics=[BinaryAccuracy(),Precision(),Recall()])
# model.fit(x_train_transformed,y_train,epochs=10,batch_size=1)

 #Initialisation weights as zero and then making model configuraion
# inital_weights=model.get_weights()
# for i in range(len(inital_weights)):
#     inital_weights[i]=np.zeros(inital_weights[i].shape)
# model.set_weights(inital_weights)
# model.compile(optimizer='adam',loss=BinaryCrossentropy(),metrics=[BinaryAccuracy(),Precision(),Recall()])
# model.fit(x_train_transformed,y_train,batch_size=5,epochs=10)

# Initialisation weights as One and then making model configuraion
# inital_weights=model.get_weights()
# print(inital_weights)
# for i in range(len(inital_weights)):
#     inital_weights[i]=np.ones(inital_weights[i].shape)
#
# model.set_weights(inital_weights)
# model.compile(optimizer='adam',loss=BinaryCrossentropy(),metrics=[BinaryAccuracy(),Precision(),Recall()])
# model.fit(x_train_transformed,y_train,batch_size=5,epochs=10)
# print(model.get_weights())

# y_pred=model.predict(x_test)
# y_pred=np.where(y_pred>0.5,1,0)
# print(y_pred)

# actual=df.loc[x_test.index]
# pred=pd.DataFrame(y_pred,index=x_test.index,columns=['Compare'])
# Overall=pd.concat([actual,pred],axis=1)
# Overall.to_csv('overall.csv')


# disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
# disp.plot()
# plot_confusion_matrix(clf,x_test_transformed,y_test)

# data.hist(figsize=(20,20))
# sb.countplot('Pregnancies',data=df,hue='Outcome')
# sb.boxplot(x=data['Pregnancies'])
# sb.scatterplot(data['Pregnancies'],data['Outcome'])
# plt.show()

# CROSS VAL SCORE
cvs=cross_val_score(kc,x_train_transformed,y_train,cv=3)
print(cvs.mean())
