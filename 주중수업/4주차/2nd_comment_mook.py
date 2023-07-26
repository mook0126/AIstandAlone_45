#필요한 패키지 임포트
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam

#하이퍼파라메터 선언
image_size = 28 # MNIST 데이터의 input Size
hidden_size = 500 # hidden_layer의 size 500으로 설정
num_classes = 10 # 출력단의 분류 class 갯수 10으로 설정
batch_size = 100 # batch_size 100
lr = 0.001 # learning rate 0.001 사용
epochs = 3 # epochs 3회 실시
device = torch.device('cuda' if torch.cuda.is_available()else'cpu') #CPU/GPU 설정

#모델 설계도 그리기
class MLP(nn.Module): 
    def __init__(self,image_size,hidden_size,num_classes): #레고 조각생성
        super().__init__() # nn.moudle를 깨우는 역할, super를 사용하여 클래스 상속해줌
        self.image_size = image_size # self를 사용하여 init 밖에서도 사용할수 있는 객체 변수로 설정
        self.mlp1 = nn.Linear(image_size*image_size,hidden_size) #첫번째 layer로 input을 받아 hidden layer로 넘김
        self.mlp2 = nn.Linear(hidden_size,hidden_size) # 2번째 layer로 hidden layer를 쌓음
        self.mlp3 = nn.Linear(hidden_size,hidden_size) # 3번째 layer로 hidden layer를 쌓음
        self.mlp4 = nn.Linear(hidden_size,num_classes) # 4번째 layer로 분류 실시
    def forward(self,x):
        # x : [batch_size,28,28,1]
        #[batch_size,28*28]
        batch_size = x.shape[0] #batch size에 따라 reshape 해줌
        x = torch.reshape(x,(-1,self.image_size*self.image_size)) # -1로 해서 batch 사이즈를 알아서 처리하게 함
        x = self.mlp1(x) #[batch_size,500]
        x = self.mlp2(x) #[batch_size,500]
        x = self.mlp3(x) #[batch_size,500]
        x = self.mlp4(x) #[batch_size,10]
        return x
    
#설계도를 바탕으로 모델을 만들어야함
myMLP = MLP(image_size,hidden_size,num_classes) #클래스를 객체로 사용하기 위해서 괄호 사용하고 파라미터들을 괄호 안에 넣음

#데이터 불러오기
#dataset설정
train_mnist = MNIST(root='../../data/mnist',train=True, transform=ToTensor(),download=True) # MNIST dataset에서 train data를 불러옴
test_mnist = MNIST(root='../../data/mnist',train=False, transform=ToTensor(),download=True) # MNIST dataset에서 test data를 불러옴
#dataloader 설정
train_loader = DataLoader(dataset = train_mnist,batch_size=batch_size,shuffle = True) # train data를 Dataloader에서 처리함
test_loader = DataLoader(dataset = test_mnist,batch_size=batch_size,shuffle = False) # test data를 Dataloader에서 처리함

#Loss 설정
loss_fn = nn.CrossEntropyLoss() # CrossEntroyLoss 사용

#Optimizer 선언
optim = Adam(params=myMLP.parameters(),lr=lr) # Adam optimizer 사용
#학습을 위한 반복(Loop) for /while
for epoch in range(epochs): # for문을 이용하여 epoch 반복
#입력할 데이터를 위해 데이터 준비
    for idx, (images,targerts) in enumerate(train_loader): #데이터를 입력
        images = images.to(device) #images 입력
        targerts = targerts.to(device) # targerts 입력
# 모델에 데이터를 넣기 
        output = myMLP(images) # 모델에 데이터 입력
# 모델의 출력과 정답을 비교하기 (Loss 사용) 
        loss = loss_fn(output,targerts) # loss_fn를 이용하여 정답과 비교
# Loss를 바탕으로 업데이트 진행 (Optimizer) 
        loss.backward() #backwrard 진행
        optim.step()  # 업데이트 진행
        optim.zero_grad() # backward 하기전에 zerograd로 없애 나야지 에러가 안남/ 기존값을 날려버려야함
# 평가(로깅, print), 저장 
        if idx % 100 == 0: #100번에 한번 loss 출력
            print(loss) #print로 loss 출력
