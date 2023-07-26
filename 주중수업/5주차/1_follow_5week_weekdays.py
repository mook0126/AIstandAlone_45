# 필요한 패키지 임포트
import os
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam

#하이퍼파라메터 선언
lr = 0.001
image_size = 28
num_classes = 10 
hidden_size = 500 
batch_size = 100 
epochs = 3 
results_folder = 'results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#저장
#상위 저장 폴더를 만듬
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

target_folder_name = max([0]+[int(e) for e in os.listdir(results_folder)])+1
target_folder =os.path.join(results_folder,str(target_folder_name))
os.makedirs(target_folder)
with open(os.path.join(target_folder,'hparam.txt'),'w') as f:
    f.write(f'{lr =}\n')
    f.write(f'{image_size =}\n')
    f.write(f'{num_classes =}\n')
    f.write(f'{hidden_size =}\n')
    f.write(f'{batch_size =}\n')
    f.write(f'{epochs =}\n')
    f.write(f'{results_folder =}\n')

#모델 설계도 그리기
class MLP(nn.Module):
    def __init__(self,image_size,hidden_size, num_classes):
        super().__init__()
        self.image_size = image_size
        self.mlp1 = nn.Linear(image_size*image_size,hidden_size)
        self.mlp2 = nn.Linear(hidden_size,hidden_size)
        self.mlp3 = nn.Linear(hidden_size,hidden_size)
        self.mlp4 = nn.Linear(hidden_size,num_classes) 

    def forward(self,x) :
        batch_size = x.shape[0]
        x = torch.reshape(x,(-1,self.image_size*self.image_size))
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        return x

#설계로를 바탕으로 모델을 만들어야함
myMLP = MLP(image_size, hidden_size, num_classes).to(device)

#데이터 불러오기
#dataset 설정
train_mnist = MNIST(root='../../data/mnist', train= True, transform=ToTensor(), download=True)
test_mnist = MNIST(root='../../data/mnist' ,train= False, transform=ToTensor(), download=True)

#dataloader 설정
train_loader = DataLoader(dataset=train_mnist,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_mnist,batch_size=batch_size,shuffle=False)

#Loss 선언
loss_fn = nn.CrossEntropyLoss()

# Optimizer 선언
optim = Adam(params =myMLP.parameters(), lr=lr)

def evaluate(model,loader,device):
    with torch.no_grad():
        model.eval()
        total = 0
        correct = 0
        for images,targets in loader:
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            output_index =torch.argmax(output,dim = 1)
            total += targets.shape[0]
            correct += (output_index == targets).sum().item()

        acc = correct/ total * 100
        model.train()
        return acc

def evaluete_by_class(model,loader,device):
    with torch.no_grad():
        model.eval()
        total = torch.zeros(num_classes)
        correct = torch.zeros(num_classes)
        for images, targets in loader:
            images,targets = images.to(device), targets.to(device)
            output = model(images)
            output_index = torch.argmax(output,dim = 1)

            for _class in range(num_classes):
                total[_class] += (targets == _class).sum().item()
                correct[_class] += ((targets == _class)* (output_index == _class)).sum().item()

        acc = correct/total * 100
        model.train()
        return acc
    
_max = -1

#학습을 위한 반복 (Loop) for / While
for epoch in range(epochs):
    for idx, (images,targets) in enumerate (train_loader):
        images = images.to(device)
        targets = targets.to(device)
        output = myMLP(images) 
        loss = loss_fn(output,targets)
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        if idx % 100 == 0:
            print(loss)
            acc = evaluate(myMLP, test_loader, device)
#            acc = evaluete_by_class(myMLP, test_loader, device,num_classes)
            if _max < acc:
                print ('새로운 acc 등장, 모델 weight 업데이트', acc)
                _max = acc
                torch.save(
                    myMLP.state_dict(),
                    os.path.join(target_folder,'myMLP_best.ckpt')
                )

 

