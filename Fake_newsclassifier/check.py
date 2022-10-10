import pandas as pd


data=pd.read_csv('train.csv')
# print(data.isnull().sum())
# new_data=data.interpolate(method="linear",axis=0)
# print(new_data.isnull().sum())

print(data.shape)
print(data.iloc[:,:-1])  # include all rows and exclude last column
print(data.iloc[:,-1])   # include all rows and last column