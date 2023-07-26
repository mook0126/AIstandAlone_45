# 필요한 패키지 임포트 
import os
import json
import torch
import torch.nn as nn 
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
import argparse


# 하이퍼파라메터 선언 -> parser
def parse_args():
    parser	=	argparse.ArgumentParser()
    #하이퍼 파라메터 설정 -> parser
    parser.add_argument('--lr', type = float, default = 0.001)  # --는 옵션을 나타내는것임
    parser.add_argument('--image_size', type = int, default = 28)  # --는 옵션을 나타내는것임
    parser.add_argument('--num_classes', type = int, default = 10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_size', type = int, default = 500)  
    parser.add_argument('--imagbatch_sizee_size', type = int, default = 100) 
    parser.add_argument('--epochs', type = int, default = 3) 
    parser.add_argument('--results_folder', type = str, default = 'results') 
    parser.add_argument('--device',  default = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) 
    parser.add_argument('--do_save',	action='store_true',help='if given,	save	results')
 #   parser.add_argument('--data', nargs='+', type=str)
                
    args = parser.parse_args()   # parse 파싱하기  // https://engineer-mole.tistory.com/213
    return args

def main() :
    args = parse_args()
    print(args.lr)
    # 저장 
    # 상위 저장 폴더를 만들어야 함 
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)
    # 내가 저장을 할 하위 폴더를 만들어야 함 (하위 폴더가 앞으로 사용될 타켓 폴더가 됨)
    target_folder_name = max ([0]+[int(e) for e in os.listdir(args.results_folder)])+1  # 아무것도 없을때 처리를 위해서 0을 넣고 아니면 최대 값에서 +1을 해서 폴더를 만들어줌
    targer_folder = os.path.join(args.results_folder,str(target_folder_name))
    os.makedirs(targer_folder)

    # 타켓 폴더 밑에 hparam 저장 (text의 형태로)
    with open(os.path.join(targer_folder,'hparam.json'),'w') as f:  # jason write https://rfriend.tistory.com/474
        write_args = args.__dict__ # device 값 처리
        del write_args['device'] # device를 제거함 https://www.freecodecamp.org/news/python-remove-key-from-dictionary/
        json.dump(args.__dict__,f,indent = 4) # dump를 통해서 hperar 가져옴 indent로 정열함
           
    assert() 

    # 모델 설계도 그리기 class MLP(nn.Module):
    class MLP(nn.Module):
        def __init__(self, image_size, hidden_size, num_classes) : # 레고 조각 생성 
            super().__init__()
            self.image_size = image_size
            self.mlp1 = nn.Linear(image_size * image_size, hidden_size) 
            self.mlp2 = nn.Linear(hidden_size, hidden_size) 
            self.mlp3 = nn.Linear(hidden_size, hidden_size) 
            self.mlp4 = nn.Linear(hidden_size, num_classes) 

        def forward(self, x): # x : [batch_size, 28, 28, 1]  # 레고 조각을 조립 
            batch_size = x.shape[0]
            x = torch.reshape(x, (-1, self.image_size * self.image_size)) # [batch_size, 28*28]
            x = self.mlp1(x) # [batch_size, 500]
            x = self.mlp2(x) # [batch_size, 500]
            x = self.mlp3(x) # [batch_size, 500]
            x = self.mlp4(x) # [batch_size, 10]
            return x 

    # 설계도를 바탕으로 모델을 만들어야 함 <- 하이퍼파라메터 사용 
    myMLP = MLP(args.image_size, args.hidden_size, args.num_classes).to(args.device)

    # 데이터 불러오기 
    # dataset 설정 
    train_mnist = MNIST(root='../../data/mnist', train=True, transform=ToTensor(), download=True)
    test_mnist = MNIST(root='../../data/mnist', train=False, transform=ToTensor(), download=True)
    # dataloader 설정 
    train_loader = DataLoader(dataset=train_mnist, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_mnist, batch_size=args.batch_size, shuffle=False)

    # Loss 선언 
    loss_fn = nn.CrossEntropyLoss()
    # Optimizer 선언 
    optim = Adam(params=myMLP.parameters(), lr=args.lr) 

    # 평가 함수 구현 
    def evaluate(model, loader, device):
        with torch.no_grad(): 
            model.eval()
            total = 0 
            correct = 0 
            for images, targets in loader: 
                images, targets = images.to(device), targets.to(device)
                output = model(images)
                output_index = torch.argmax(output, dim =1)
                total += targets.shape[0]
                correct +=  (output_index == targets).sum().item()
            
            acc = correct / total *100
            model.train()
            return acc

    def evaluete_by_class(model,loader,device,num_classes):
        with torch.no_grad(): 
            model.eval()
            correct = torch.zeros(num_classes) # class를 구별하기 위해서 10개의 zero 값으로 세팅함
            total = torch.zeros(num_classes)
            for images, targets in loader: 
                images, targets = images.to(device), targets.to(device)
                output = model(images)
                output_index = torch.argmax(output, dim =1)
                
                for _class in range(num_classes): # 모든 class에 대해서 계산하기 위해 for문을 사용함
                    total[_class] += (targets == _class).sum().item()
                    correct[_class] += ((targets == _class) * (output_index== _class)).sum().item() # and 연산은 *로 처리함

    #            total += targets.shape[0]
    #            correct +=  (output_index == targets).sum().item()
            
            acc = correct/total * 100 #각 class에 대한 acc를 구할수 있다.
            model.train()
            return acc
    _max = -1
    # 학습을 위한 반복 (Loop) for / while 
    for epoch in range(args.epochs): 
    # 입력할 데이터를 위해 데이터 준비 (dataloader) 
        for idx, (images, targets) in enumerate(train_loader):
            images = images.to(args.device)
            targets = targets.to(args.device)

            # 모델에 데이터를 넣기 
            output = myMLP(images)
            # 모델의 출력과 정답을 비교하기 (Loss 사용) 
            loss = loss_fn(output, targets)
            # Loss를 바탕으로 업데이트 진행 (Optimizer) 
            loss.backward()
            optim.step()
            optim.zero_grad()

            if idx % 100 == 0: 
                print(loss)
            # 평가(로깅, print), 저장
                acc = evaluate(myMLP, test_loader, args.device)
    #            acc = evaluete_by_class(myMLP, test_loader, device,num_classes)
                # 평가 결과가 좋으면 타켓 폴더에 모델 weight 저장을 진행
                # 평가 결과가 좋다는게 무슨 의지미? ->과거의 평가 결과보다 좋은 수치가 나오면 결과가 좋다고 함
                # 과거 결과(max) < 지금 결과(acc)
                if _max < acc :
                    print ('새로운 acc 등장, 모델 weight 업데이트', acc )
                    _max = acc
                    torch.save(
                        myMLP.state_dict(),
                        os.path.join(targer_folder,'myMLP_best.ckpt')
                    )

if __name__ =='__main__':
    main()