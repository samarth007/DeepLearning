import pandas_datareader as pdr
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

data=pdr.get_data_yahoo('TSLA')
# data['High'].plot(xlim=['2021-04-24','2021-07-23'],figsize=(12,4))
# data['High'].resample(rule='A').max().plot(kind='bar',figsize=(10,4))
# print(data.resample(rule='BA').min())
data['Open:30 days']=data['Open'].rolling(window=30,min_periods=1).mean()
data['Open:50 days']=data['Open'].rolling(window=50,min_periods=1).mean()

# data[['Open','Open:30 days','Open:50 days']].plot(xlim=['2020-01-01','2021-01-01'],figsize=(12,4))
# plt.show()
