import numpy as np


arr=[[0,1,2,3],[45,6,7,8],[10,23,3,2],[5,24,23,7]]
ar=np.array(arr)
print(ar)
print(np.argmax(ar))
print(np.argmax(ar,axis=1))
print(np.argmax(ar,axis=0))

# x = np.arange(20).reshape(4,5) + 7
# y=np.argmax(x, axis=0)
# z=np.argmax(x, axis=1)
# print(x)
# print(y)
# print(z)