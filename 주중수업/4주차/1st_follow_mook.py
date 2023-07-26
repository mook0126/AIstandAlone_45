#필요한 패키지 임포트
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam

#하이퍼파라메터 선언
image_size = 28
hidden_size = 500
num_classes = 10
batch_size = 100
lr = 0.001
epochs = 3
device = torch.device('cuda' if torch.cuda.is_available()else'cpu')                     

#모델 설계도 그리기
class MLP(nn.Module):
    def __init__(self,image_size,hidden_size,num_classes): #레고 조각생성
        super().__init__()
        self.image_size = image_size
        self.mlp1 = nn.Linear(image_size*image_size,hidden_size)
        self.mlp2 = nn.Linear(hidden_size,hidden_size)
        self.mlp3 = nn.Linear(hidden_size,hidden_size) 
        self.mlp4 = nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        # x : [batch_size,28*28,1]
        batch_size = x.shape[0]
        x = torch.reshape(x,(-1,self.image_size*self.image_size))
        x = self.mlp1(x) #[batch_size,500]
        x = self.mlp2(x) #[batch_size,500]
        x = self.mlp3(x) #[batch_size,500]
        x = self.mlp4(x) #[batch_size,10]
        return x
    
#설계도를 바탕으로 모델을 만들어야함
myMLP = MLP(image_size,hidden_size,num_classes)

#데이터 불러오기
#dataset설정
train_mnist = MNIST(root='../../data/mnist',train=True, transform=ToTensor(),download=True)
test_mnist = MNIST(root='../../data/mnist',train=False, transform=ToTensor(),download=True)
#dataloader 설정
train_loader = DataLoader(dataset = train_mnist,batch_size=batch_size,shuffle = True)
test_loader = DataLoader(dataset = test_mnist,batch_size=batch_size,shuffle = False)

#Loss 설정
loss_fn = nn.CrossEntropyLoss()

#Optimizer 선언
optim = Adam(params=myMLP.parameters(),lr=lr)
#학습을 위한 반복(Loop) for /while
for epoch in range(epochs):
#입력할 데이터를 위해 데이터 준비
    for idx, (images,targerts) in enumerate(train_loader):
        images = images.to(device)
        targerts = targerts.to(device)
# 모델에 데이터를 넣기 
        output = myMLP(images)
# 모델의 출력과 정답을 비교하기 (Loss 사용) 
        loss = loss_fn(output,targerts)
# Loss를 바탕으로 업데이트 진행 (Optimizer) 
        loss.backward()
        optim.step()
        optim.zero_grad()

        if idx % 100 == 0:
            print(loss)
# 평가(로깅, print), 저장 
