import time
import torch 
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

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

class Convet(nn.Module):
    def __init__(self):
        super(Convet,self).__init__()
        self.conv1=nn.Conv2d(3,10,5,stride=2)   #100,3,32,32input    100,10,14,14 output
        self.pool=nn.MaxPool2d(2,2)            #100,10,14,14 input   100,10,7,7 output
        self.conv2=nn.Conv2d(10,20,5,stride=2)  #100,10,7,7input 100,20,2,2 output
        self.fc1=nn.Linear(20*1*1,120)          #flattening 
        self.fc2=nn.Linear(120,80)
        self.fc3=nn.Linear(80,10)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)        #100,20,2,2 input  100,20,1,1 output
        x=torch.flatten(x,1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

model=Convet().to(device)
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learn)

n_total_step=len(train_loader)
for epoch in range(epochs):
    running_loss=0
    for j, (image,label) in enumerate(train_loader):
        image=image.to(device)  #torch.Size([100, 3, 32, 32])
        label=label.to(device)  #torch.Size([100])
        start=time.time()
        y_pred=model(image)
        l=loss(y_pred,label)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()  
        running_loss+=l.item()
    print(f'[{epoch +1}] loss: {running_loss/n_total_step}')    

PATH='./TorchPractice/cnn.pth'
torch.save(model.state_dict(),PATH)


# loaded_model=Convet()
# loaded_model.load_state_dict(torch.load(PATH))
# loaded_model.to(device)
# loaded_model.eval()

# with torch.no_grad():
#     n_correct=0
#     n_correct2=0
#     n_sample=len(test_loader)

#     for images,label in test_loader:
#         images=images.to(device)
#         label=label.to(device)
#         output=model(images)

#         _, predicted=torch.max(output,1)
#         n_correct+=(predicted==label).sum().item()
        
#         output2=loaded_model(images)
#         _,predicted2=torch.max(output2,1)
#         n_correct2=(predicted2==label).sum().item()

#     acc=n_correct/n_sample * 100
#     print('Acuracy with original model ',acc)

#     acc1=n_correct2/n_sample * 100
#     print('Accuracy with saved model ',acc1)