from keras_preprocessing.text import one_hot
from tensorflow.python.keras.layers import Embedding
from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
import numpy as np


sentence=['the glass of milk','the glass of coffee','krish naik youtube channel','trip to goa',
          'aspiring data scientist','being human','java developer with spring as skillset']
label=[1,1,0,0,1,1,0]
voc_size=5000

#One hot encoding
onehot_repo= [one_hot(word,voc_size) for word in sentence]
print(onehot_repo)
#Pre Padding
sent_length=8
embedded_docs=pad_sequences(onehot_repo,padding='pre',maxlen=sent_length)
print(embedded_docs)
X=embedded_docs
Y=np.array(label)
dim=20
model=Sequential()
model.add(Embedding(voc_size,dim,input_length=sent_length))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X,Y,epochs=1)
