# Debug mode를 이용해 상해있는 이 코드를 정상적으로 작동시켜보세요. 
# 총 11가지의 강제 error를 만들었습니다. 
# 에러를 고치면 에러가 발생한 line 뒤쪽에 주석으로 error의 원인을 적어주세요. 
# Debug mode의 call stack, debug console, variable 등의 과정을 충분히 활용해보세요 ^^ 
# 제출은 고친 파일(error의 원인이 적혀있는)과 
# debug mode를 활용해 디버깅하는 과정의 스크린샷을 찍어서 보내주세요!

import torch  
import torch.nn as nn 
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

lr = 0.001 
image_size = 28 
num_classes = 10 
batch_size = 100
hidden_size = 500 
total_epochs = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda & cpu 자리 수정

class MLP(nn.Module): 
    def __init__(self, image_size, hidden_size, num_classes) : 
        self.image_size = image_size
        super().__init__()
        self.mlp1 = nn.Linear(in_features=image_size*image_size, out_features=hidden_size) # mlp1 모듈을 제대로 불러오지 못함 nn 모듈을 사용하기 위해 super().__init__()으로 모듈을 깨워야함
        self.mlp2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp4 = nn.Linear(in_features=hidden_size, out_features=num_classes)
    
    def forward(self, x) : 
        batch_size = x.shape[0]
        x = torch.reshape(x, ( -1, self.image_size * self.image_size))
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)  # mat1 and mat2 shapes cannot be multiplied (100x10 and 500x10)  // 500이 입력으로 들어오는데 10이 출력으로 받는 모델로 사이즈가 안맞음 // 전 layer의 out_feature 사이즈 수정
        return x

myMLP = MLP(image_size, hidden_size, num_classes)

train_mnist = MNIST(root='../../data/mnist', train=True, transform=ToTensor(), download=True)
test_mnist = MNIST(root='../../data/mnist', train=False, transform=ToTensor(), download=True)


train_loader = DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True)   #batch_size should be a positive integer value, but got batch_size=batch_size // batch_size가 integer가 아닌 string으로 되어 있어서 inter 값인 hyperparameter값을 가져오게 수정
test_loader = DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss() # 클래스 객체화

optim = Adam(params=myMLP.parameters(), lr=lr) #'<=' not supported between instances of 'float' and 'str' //float 를 지원하지 않기 때문에 lr를 int형으로 변환
                                            # 'method' object is not iterable // params 값을 제대로 불러오지 못함 클래서 형식으로 변경


for epoch in range(total_epochs): 
    for idx, (image, label) in enumerate(train_loader) : 
        image = image.to(device)  # Torch not compiled with CUDA enabled // cuda 지원을 안하기 때문에 cpu가 진행 되도록 수정
        label = label.to(device)

        output = myMLP(image)

        loss = loss_fn(output, label)  #Boolean value of Tensor with more than one value is ambiguous // nn.crossEntropyLoss 클래스화
                                        #Expected target size [100, 10], got [100] // output 차원이 맞지 않음


        loss.backward()
        optim.step() # 'Tensor' object has no attribute 'step' // step() 사용하기 위해 loss가 아닌 optim에서 불러옴

        optim.zero_grad()

        if idx // 100 == 0 : 
            print(loss)

