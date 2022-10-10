import torch
import torch.nn as nn

class cnn(nn.Module):
    def __init__(self):
        super(cnn,self).__init__()
        self.c=nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=2,padding=2)
        print(self.c)
model=cnn()   
for p in model.parameters():
    print(p)     