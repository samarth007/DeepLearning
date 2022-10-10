
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder

#
# # generate some data
# X, y = make_classification(n_classes=5, n_samples=5420, n_features=212, n_informative=212, n_redundant=0, random_state=42)
# print(X.shape, y.shape)
# # (5420, 212) (5420,)
#
# # one-hot encode the target
# Y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
# print(X.shape,Y.shape)
# # (5420, 212) (5420, 5)
#
# # extract the input and output shapes
# input_shape = X.shape[1]
# output_shape = Y.shape[1]
# print(input_shape, output_shape)
# # 212 5
#
# # define the model
# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(output_shape, activation='softmax'))
#
# # compile the model
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # fit the model
# history = model.fit(X, Y, epochs=3)

l=[1,2,3,4]
ll=[4,5,6]
l.append(1)
l.append(3)
l.append(ll)
l.extend(ll)
print(l)