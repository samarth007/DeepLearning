import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import Embedding,Dense,LSTM,RNN
from tensorflow.python.keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import one_hot
from sklearn.model_selection import train_test_split
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
# nltk.download('stopwords')
import re

data=pd.read_csv(r'D:\\DeepLearning\\Fake_newsclassifier\\train.csv')
data=data.dropna()
X=data.drop('label',axis=1)
Y=data['label']

voc_size=5000
message=X.copy()
message.reset_index(inplace=True)
corpus=[]
lem=WordNetLemmatizer()
for i in range(len(message)):
    review=re.sub('[^a-zA-Z]',' ',message['title'][i])
    review=review.lower()
    review=review.split()
    review=[lem.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

onehot_repo=[one_hot(word,voc_size) for word in corpus]
sent_length=20
embedded_docs=pad_sequences(onehot_repo,padding='pre',maxlen=sent_length)
vector_feature=40
model=Sequential()
model.add(Embedding(voc_size,vector_feature,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

X_final=np.array(embedded_docs)
Y_final=np.array(Y)

x_train,x_test,y_train,y_test=train_test_split(X_final,Y_final,test_size=0.3,random_state=42)
model.fit(x_train,y_train,epochs=10,batch_size=30)
