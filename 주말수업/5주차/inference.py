# 패키지 import
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
# 타켓하는 학습 세팅을 설정 
target_folder = '../../주중수업/5주차/results/13'
assert os.path.exists(target_folder),'target folder doesnt exits' # 강제로 에러를 만들어 코드를 끊는 코드
print('hello')

# 하이퍼파라메터 로드
with open(os.path.join(target_folder,'hparam.txt'),'r')as f:
    data = f.readlines()

lr =float(data[0].strip())
image_size =int(data[1].strip()) 
num_classes =int(data[2].strip())  
hidden_size =int(data[3].strip())  
batch_size =int(data[4].strip())  
epochs =int(data[5].strip())  
results_folder =data[6].strip()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 모델 class 만들기
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

# 모델 객체 만들기
myMLP = MLP(image_size, hidden_size, num_classes).to(device)

# 모델 weight를 업데이트
ckpt = torch.load(
         os.path.join(
            target_folder,'myMLP_best.ckpt'
            )
        )

myMLP.load_state_dict(ckpt)



# 추론 데이터를 가지고오기
image_path = './test_image.jpg'
assert os.path.exists(image_path), 'target image doesnt exists'

input_image = Image.open('./test_image.jpg').convert("L") # PIL import 해서 image open
# 학습 과정에서 사용했떤 전처리 과정을 그래도 실행
# 크기 맞추기 -> tensor로 만들기
resizer = Resize(image_size) #사이즈 변환
totensor = ToTensor ()  # tensor로 변환
image = totensor(resizer(input_image)) # PIL 파일을 tensor로 변환
# 
# 모델 추론 진행
output = myMLP(image)  
output = torch.argmax(output).item()

print(f'model says, the image is {output}')

# 추론 결과를 우리가 이해할수 있는 형태로 변환
