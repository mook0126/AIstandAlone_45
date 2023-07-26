#필요한 패키지 임포트
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


#하이퍼파라메터 선언
input_size = 28 # MNIST 데이터의 input Size
hidden_layer = 500 # hidden_layer의 size 500으로 설정
class_nums = 10 # 출력단의 분류 class 갯수 10으로 설정
batch_size = 100 # batch_size 100
lr = 0.001 # learning rate 0.001 사용
epochs = 3 # epochs 3회 실시
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #CPU/GPU 설정

#모델 설계도 그리기
class MLP(nn.Module):
    def __init__(self,input_size,hidden_layer,class_nums):
        super().__init__()
        self.input_size = input_size
        self.mlp1 = nn.Linear(input_size*input_size,hidden_layer)
        self.mlp2 = nn.Linear(hidden_layer,hidden_layer)
        self.mlp3 = nn.Linear(hidden_layer,hidden_layer)
        self.mlp4 = nn.Linear(hidden_layer,class_nums)
    def forward(self,x):
        batch_size = x.shape[0]
        x = torch.reshape(x,(-1,self.input_size*self.input_size))
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        return x

 #nn.moule class 생성
 #레고 조각생성
 # nn.moudle를 깨우는 역할, super를 사용하여 클래스 상속해줌
 # self를 사용하여 init 밖에서도 사용할수 있는 객체 변수로 설정
 # 첫번째 layer로 input을 받아 hidden layer로 넘김
 # 2번째 layer로 hidden layer를 쌓음
 # 3번째 layer로 hidden layer를 쌓음
 # 4번째 layer로 분류 실시
 # 레고 조각을 조립 
        # x : [batch_size,28,28,1]
        #[batch_size,28*28]
        #batch size에 따라 reshape 해줌
         # -1로 해서 batch 사이즈를 알아서 처리하게 함
         #[batch_size,500]
         #[batch_size,500]
         #[batch_size,500]
         #[batch_size,10]
     
    
#설계도를 바탕으로 모델을 만들어야함
 #클래스를 객체로 사용하기 위해서 괄호 사용하고 파라미터들을 괄호 안에 넣음
myMLP = MLP(input_size,hidden_layer,class_nums).to(device)
#데이터 불러오기
train_MNIST = MNIST(root= '../../data/MNIST',train=True,transform=ToTensor(),download= True)
test_MNIST = MNIST(root= '../../data/MNIST',train=False,transform=ToTensor(),download= True)
#dataset설정
 # MNIST dataset에서 train data를 불러옴
 # MNIST dataset에서 test data를 불러옴
train_DataLoader = DataLoader(dataset= train_MNIST,batch_size=batch_size,shuffle= True)
test_DataLoader = DataLoader(dataset= test_MNIST,batch_size=batch_size,shuffle= False)
#dataloader 설정
 # train data를 Dataloader에서 처리함
 # test data를 Dataloader에서 처리함

#Loss 설정
 # CrossEntroyLoss 사용
loss_fn = nn.CrossEntropyLoss()
#Optimizer 선언
 # Adam optimizer 사용
optim = torch.optim.Adam(params=myMLP.parameters(),lr=lr)
#학습을 위한 반복(Loop) for /while
 # for문을 이용하여 epoch 반복
for epoch in range(epochs):
    for idx, (images,targets) in enumerate (train_DataLoader):
        images = images.to(device)
        targets = targets.to(device)
        output = myMLP(images)
        loss = loss_fn(output,targets)
        loss.backward()
        optim.step()
        optim.zero_grad()

        if idx % 100 == 0:
            print(loss)

 #입력할 데이터를 위해 데이터 준비
     #데이터를 입력
         #images 입력
         # targerts 입력
# 모델에 데이터를 넣기 
        # 모델에 데이터 입력
# 모델의 출력과 정답을 비교하기 (Loss 사용) 
       # loss_fn를 이용하여 정답과 비교
# Loss를 바탕으로 업데이트 진행 (Optimizer) 
         #backwrard 진행
        # 업데이트 진행
         # backward 하기전에 zerograd로 없애 나야지 에러가 안남/ 기존값을 날려버려야함
# 평가(로깅, print), 저장 

        #100번에 한번 loss 출력
             #print로 loss 출력

