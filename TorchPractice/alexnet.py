from operator import mod
import torch
import torch.nn as nn
import torchvision.transforms as trans
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

batch=1000
num_cls=10
epochs=5
learn=0.01
t=trans.Compose([trans.Resize((227,227)),trans.ToTensor()])

train_dataset=torchvision.datasets.CIFAR10(root='./TorchPractice/data',train=True,transform=t)
test_dataset=torchvision.datasets.CIFAR10(root='./TorchPractice/data',train=False,transform=t)

train_loader=DataLoader(dataset=train_dataset,batch_size=batch,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch,shuffle=False)

class Allnet(nn.Module):
    def __init__(self):
        super(Allnet,self).__init__()
        self.l1=nn.Conv2d(3,96,11,stride=(4,4),padding=0)
        self.pool=nn.MaxPool2d(kernel_size=3,stride=2)
        self.l2=nn.Conv2d(96,256,5,stride=1,padding=0)
        self.l3=nn.Conv2d(256,384,3,stride=1,padding=0)
        self.l4=nn.Conv2d(384,384,3,stride=1,padding=0)
        self.l5=nn.Conv2d(384,256,3,stride=1,padding=0)
        self.fc1=nn.Linear(256*2*2,2000)
        self.fc2=nn.Linear(2000,1024)
        self.fc3=nn.Linear(1024,10)

    def forward(self,x):
        x=F.relu(self.l1(x))
        x=self.pool(x)
        x=F.relu(self.l2(x))
        x=self.pool(x)
        x=F.relu(self.l3(x))
        x=F.relu(self.l4(x))
        x=F.relu(self.l5(x))
        x=self.pool(x)
       # print(x.shape)
        x=torch.flatten(x,1)
        #print(x.shape)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x
            
model=Allnet()

learn=0.01
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learn)

n_step=len(train_loader)
for epoch in range(epochs):
    running_loss=0
    for image,labels in train_loader:
        output=model(image)
        l=loss(output,labels)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss+=l.item()
    print(f'[{epoch +1}] loss: {running_loss/n_step}')
