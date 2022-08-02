import numpy as np
import re
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from keras_preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences

sentence=['Life: Life Of Luxury: Elton Johnâ€™s 6 ' \
         'Favorite Shark Pictures To Stare At During Long, Transcontinental Flights',
          'A Back-Channel Plan for Ukraine and Russia, Courtesy of Trump Associates - The New York Times']

voc_size=1000
lm=WordNetLemmatizer()
corpus=[]
for i in range(len(sentence)):
   review=re.sub('[^a-zA-Z]',' ',sentence[i])
   print(review)
   review=review.lower()
   print(review)
   review=review.split()
   print(review)
   review=[lm.lemmatize(word) for word in review if word not in stopwords.words('english')]
   print(review)
   review=' '.join(review)
   print(review)
   corpus.append(review)

print(corpus)
onehot_repo=[one_hot(word,voc_size) for word in corpus]
print(onehot_repo)
sent_length=20
embedded_docs=pad_sequences(onehot_repo,padding='pre',maxlen=sent_length)
print(embedded_docs)

# two_shape=(4,2)
# three_shape=(2,3,2)
# input=np.ones(three_shape)
# print(input)