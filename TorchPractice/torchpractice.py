import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time


batch=100
num_cls=10
epochs=5
learn=0.01

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_dataset=torchvision.datasets.CIFAR10(root='./TorchPractice/data',train=True,transform=trans)
test_dataset=torchvision.datasets.CIFAR10(root='./TorchPractice/data',train=False,transform=trans)

train_loader=DataLoader(dataset=train_dataset,shuffle=True,batch_size=batch)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch,shuffle=False)

classes={'plane','car','bird','cat','deer','dog','frog','horse','ship','truck'}

class convetCNN(nn.Module):
    def __init__(self):
        super(convetCNN,self).__init__()
        self.conv1=nn.Conv2d(3,10,5,stride=1,padding=2)   #100,3,32,32input    100,10,32,32 output
        self.pool=nn.MaxPool2d(2,2)                     #100,10,32,32 input   100,10,16,16 output
        self.conv2=nn.Conv2d(10,20,5,stride=1,padding=2)  #100,10,16,16input 100,20,16,16 output
        self.fc1=nn.Linear(20*8*8,120)          #flattening 
        self.fc2=nn.Linear(120,80)
        self.fc3=nn.Linear(80,10)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)        #100,20,16,16 input  100,20,8,8 output
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

model=convetCNN()
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learn)

n_total_step=len(train_loader)
for epoch in range(epochs):
    running_loss=0
    for images,labels in train_loader:
        y_pred=model(images)
        l=loss(y_pred,labels)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()  
        running_loss+=l.item()
    print(f'[{epoch +1}] loss: {running_loss/n_total_step}')  

