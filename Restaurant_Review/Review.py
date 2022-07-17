import re
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from keras_preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Embedding,Dense,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data=pd.read_csv('Chrome_Review.csv')

data=data.drop(['Developer Reply','Version'],axis=1)
new_data=data.dropna(axis=0)
new_data.reset_index(drop=True,inplace=True)

output_class=len(new_data['Star'].unique())
Y=pd.get_dummies(new_data['Star'])

lem=WordNetLemmatizer()
corpus=[]
for i in range(len(new_data)):
    review=re.sub("[^a-zA-Z]",' ',str(new_data['Text'][i]))
    review=review.lower()
    review=review.split()
    review=[lem.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

voc_size=1000
oneHotRepo=[one_hot(wrd,voc_size) for wrd in corpus]
sentence_length=10
embeded_docs=pad_sequences(oneHotRepo,padding='pre',maxlen=sentence_length)
X=embeded_docs

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
y_test=np.array(y_test)
vector_dimension=50
cb=EarlyStopping(monitor='loss',min_delta=1,patience=10,mode='min')
model=Sequential()
model.add(Embedding(voc_size,vector_dimension,input_length=sentence_length))
model.add(LSTM(100))
model.add(Dense(output_class,kernel_initializer='glorot_uniform',activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=80)
y_pred=model.predict(x_test)

y_pred_arg=np.argmax(y_pred,axis=-1) #for multi-class classification
y_test_class=np.argmax(y_test,axis=1) #converted y_test from dataframe to ndarray
# y_pred_cls=model.predict_classes(x_test)  #depricated instead use above argmax (bookmark)

cr=confusion_matrix(y_pred_arg,y_test_class)
print(cr)