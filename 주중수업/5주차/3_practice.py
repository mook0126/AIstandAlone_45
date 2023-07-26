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
target_folder_name = max([0]+[int(e) for e in os.listdir(results_folder)])+1  #hparm을 저장하기 위한 하위 폴더 이름 생성
target_folder =os.path.join(results_folder,str(target_folder_name)) # 폴더 생성
os.makedirs(target_folder)
with open(os.path.join(target_folder,'hparam.txt'),'w') as f: #파리미터 값 저장
    f.write(f'{lr =}\n')
    f.write(f'{image_size =}\n')
    f.write(f'{num_classes =}\n')
    f.write(f'{hidden_size =}\n')
    f.write(f'{batch_size =}\n')
    f.write(f'{epochs =}\n')
    f.write(f'{results_folder =}\n')

#hparm을 저장하기 위한 하위 폴더 이름 생성
 # 폴더 생성

# 타켓 폴더 밑에 hparam 저장 (text의 형태로)
 #파리미터 값 저장
  

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

def evaluate(model,loader,device):
    with torch.no_grad():
        model.eval()
        total = 0
        correct = 0
        for images,targets in loader:
            images , targets = images.to(device),targets.to(device)
            output = model(images)
            output_index = torch.argmax(output,dim =1)
            total += targets.shape[0]
            correct += (output_index == targets).sum().item()
        acc = correct/total
        model.train()
        return acc    
            
# evaluate 함수 선언
    # no_grad 업데이트
       # eval 모드로 진입
        # total 갯수 초기화
        # correct 갯수 초기화
        # for문을 사용하여 images와 target을 loader에서 가져옴
            # dataloader에서 image와 target를 하나 씩 꺼내옴
            # image에 대하여 model 를 돌려 output 뽑음
             # output 결과에 대한 정답을 argmax를 통하여 가장 확률 높은것을 가져옴
             # 총 target 갯수에 대하여 totlal 갯수로 카운트 
              # output 결과가 targert과 일치하는 값을 갯수를 새줌, item()을 이용하여 tensor를 int로 뽑아냄
            
         # acc 뽑아냄 acc = correct / total 사용
          # train 모드로 들어감
          # acc 값을 return해줌


def evaluate_class (model,loader,device):
    with torch.no_grad():
        total = torch.zeros(num_classes)
        correct = torch.zeros(num_classes)
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            output_index = torch.argmax(output, dim = 1)
            for _class in range(num_classes):
                total[_class] += (targets == _class).sum().item()
                correct[_class] += ((targets ==_class)*(output_index == _class)).sum().item()
                print()
            acc = correct/total * 100
            model.train()
            return acc
_max = -1
            
# evaluate_class 함수 선언
    # no grad 업데이트
        # eval 모드로 진입
        # class num에 맞게 zeros tensor total 만듬
        # class num에 맞게 zeros tensor correct 만듬
        # dataloader에서 image와 target를 하나씩 꺼내옴
        # for문을 사용하여 images와 target을 loader에서 가져옴
            # dataloader에서 image와 target를 하나 씩 꺼내옴
            # image에 대하여 model 를 돌려 output 뽑음
            # output 결과에 대한 정답을 argmax를 통하여 가장 확률 높은것을 가져옴  
                # 각 클래스 별로 카운트를 위한 for 문
                    # 각 class 갯수를 count 
                    # output값과 정답이 맞는것을 class 별로 count
               

          # acc 뽑아냄 acc = correct / total 사용
         # train 모드로 들어감
        # acc 값을 return해줌
 # weight 최신 값을 -1로 초기화
    

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

#평가

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
            acc = evaluate(myMLP,test_loader,device)# evaluate 하여 acc 값에 저장
            acc_classs = evaluate_class(myMLP,test_loader,device)
            # class 별 evaluate 하여 acc 값에 저장
        # 평가 결과가 좋으면 타켓 폴더에 모델 weight 저장을 진행
        # 평가 결과가 좋다는게 무슨 의지미? ->과거의 평가 결과보다 좋은 수치가 나오면 결과가 좋다고 함

            if _max < acc :
                print('Find new acc and update model weight',acc)
                _max = acc
                torch.save(
                    myMLP.state_dict(),
                    os.path.join(target_folder,'myMLP_best.ckpt')
                )     
                 # 과거 결과(max) < 지금 결과(acc)         
                 # acc 저장 내역 print  출력
                  # 최신 acc를 _max에 저장
                 # 모델 값 저장
                   