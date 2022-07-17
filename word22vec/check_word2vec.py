import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# wv=api.load('word2vec-google-news-300') it will take 1.6gb to load google pre-trained model


data=pd.read_csv('test.csv',sep='\t',names=['label','message'])
data=data.dropna(axis=0)
Y=pd.get_dummies(data['label'],drop_first=True)

corpus=[]
lem=WordNetLemmatizer()
for i in range(len(data)):
    review=re.sub('[^a-zA-Z]',' ',data['message'][i])
    review=review.lower()
    review=review.split()
    review=[lem.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

words=[]   # converting each sentence into word of list and appending it to list
for sent in corpus:
    sent_token=sent_tokenize(sent)
    for wrd in sent_token:
        words.append(simple_preprocess(wrd))

model=Word2Vec(words,window=8,min_count=1,vector_size=100)
# print(model.wv.index_to_key) # returns list of vocublary
print(model.wv['happiness']) #returns vector dimension for word in dataset
print(model.wv.most_similar('happiness'))

# Taking mean of vector for each sentences
def avg_word_2_vec(docs):
    return np.mean([model.wv[w] for w in docs if w in model.wv.index_to_key],axis=0)

X_new=[]  #appending mean of vector of each sentences
for i in range(len(words)):
    X_new.append(avg_word_2_vec(words[i]))

X_aray=np.array(X_new)
#
# x_train,x_test,y_train,y_test=train_test_split(X_aray,Y,test_size=0.3,random_state=42)
# gb=GaussianNB()
# gb.fit(x_train,y_train)


#################Incomplete####################################