import numpy as np

y=[]
arr=['dog','cat','cat','dog','dog']
convert=lambda ss: int(ss=='dog')
for i in arr:
 categ=convert(i)
 y.append(categ)
print(y)

X=np.random.randint(2,10,(4,5))
print(X)
S=np.array(X).reshape(10,2)
print(S)