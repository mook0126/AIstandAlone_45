# 필요한 패키지 임포트
import torch
import torch.nn as nn
from torchvision.datasets import MNIST # MNIST import
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#모델 설계도 그리기
class MLP(nn.Module):
    def __init__(self,image_size,hidden_size, num_classes): #레고 조각 생성 # super 는 상속해 주는 대상을 깨우는 방식 / 부모를 객체로 만들고 초기화 하는 코드
        super().__init__() # nn.Module를 깨우는 역할, super를 사용하여 클래스 상속해줌
        self.image_size = image_size #self를 사용하여 init 밖에서 사용할수 있는 객체 변수로 사용할수 있도록 해줌 
        self.mlp1 = nn.Linear(image_size*image_size,hidden_size)
        self.mlp2 = nn.Linear(hidden_size,hidden_size)
        self.mlp3 = nn.Linear(hidden_size,hidden_size)
        self.mlp4 = nn.Linear(hidden_size,num_classes) 

    def forward(self,x) :
        # x : [batch_size, 28,28,1]
        #[batch_size, 28*28]
        batch_size = x.shape[0] # batch size에 따라 reshape한다.
        x = torch.reshape(x,(-1,self.image_size*self.image_size)) # -1로 해서 batch 사이즈를 알아서 처리하게 함
        x = self.mlp1(x) #[batch_size, 500]
        x = self.mlp2(x) #[batch_size, 500]
        x = self.mlp3(x) #[batch_size, 500]
        x = self.mlp4(x) #[batch_size, 10]
        return x

#설계로를 바탕으로 모델을 만들어야함
myMLP = MLP(image_size, hidden_size, num_classes).to(device) #클래스를 객체로 사용하기 위해서 괄호 사용하고 파라미터들을 괄호 안에 넣음

#데이터 불러오기
#dataset 설정
train_mnist = MNIST(root='../../data/mnist', train= True, transform=ToTensor(), download=True) # MNIIST 데이터를 불러오기 , dafualt값이 없는 애들은 설정을 해줘야 한다. 
# root = 데이터가 다운로드 되는 위치 (.. 상위 폴더) /  Train = True 학습용, False 평가용 / transform = 데이터를 가져 올때 전처리 방식 (MNIST는 정규화만 진행함), 
# Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
test_mnist = MNIST(root='../../data/mnist' ,train= False, transform=ToTensor(), download=True)

#dataloader 설정
train_loader = DataLoader(dataset=train_mnist,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_mnist,batch_size=batch_size,shuffle=False)

#Loss 선언
loss_fn = nn.CrossEntropyLoss()
    #CrossEntropy 사용

# Optimizer 선언
optim = Adam(params =myMLP.parameters(), lr=lr)
    #Adam optimizer를 사용하며, params에서 w 값들을 써줘야함.myMLP를 통해서 접근함 / myMLP.parameters() nn.mooudle에 정의되어 있으며 모델의 모든 파라미터를 가져옴
    # mlp3,mp4만 업데이트하는경우
    # optim = Adam(params = [myMLP.mlp3.parameters(), myMLP.mlp4.parameters()], lr=lr)


#학습을 위한 반복 (Loop) for / While
for epoch in range(epochs):
    for idx, (images,targets) in enumerate (train_loader):
        images = images.to(device)
        targets = targets.to(device)
        output = myMLP(images) 
        loss = loss_fn(output,targets)
        loss.backward()
        optim.step()
        optim.zero_grad() # backward 하기전에 zerograd로 없애 나야지 에러가 안남/ 기존값을 날려버려야함
        
        if idx % 100 == 0:
            print(loss)
