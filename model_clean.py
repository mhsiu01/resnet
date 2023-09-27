import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride, no_skip):
        super().__init__()
        assert int(out_planes/in_planes)==stride
        
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        
        self.no_skip = no_skip
        self.skip = nn.Sequential()
        if stride!=1 and not no_skip:
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.skip(x)) if not self.no_skip else F.relu(out)
        return out
        
def make_layer(n, in_planes, out_planes, stride, no_skip):
    blocks = [Block(in_planes, out_planes, stride=stride, no_skip=no_skip)]
    for i in range(1,n):
        blocks.append(Block(out_planes, out_planes, stride=1, no_skip=no_skip))
    return nn.Sequential(*blocks)
        
class ResNet_cifar(nn.Module):
    def __init__(self, conf=None, metrics=None, planes=[16, 16, 32, 64], strides=[1,2,2]):
        super().__init__()
        self.k = conf['k']
        planes = [p*conf['k'] for p in planes]
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, planes[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes[0])
        )
        self.layer1 = make_layer(conf['n'], planes[0], planes[1], strides[0], conf['no_skip'])
        self.layer2 = make_layer(conf['n'], planes[1], planes[2], strides[1], conf['no_skip'])
        self.layer3 = make_layer(conf['n'], planes[2], planes[3], strides[2], conf['no_skip'])
        self.layer4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(8,8)),
            nn.Flatten(1,-1),
            nn.Linear(planes[3], conf['num_classes'])
        )
        self.conf = conf
        self.metrics = metrics

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import _LRScheduler    
    
class BatchScaleLR(_LRScheduler):
    # Helpful: https://detectron2.readthedocs.io/en/latest/_modules/detectron2/solver/lr_scheduler.html
    def __init__(self, optimizer, k, last_epoch=-1):
        self.optimizer = optimizer
        self.k = k # LR scale factor
        self.warmup = 5
        super().__init__(optimizer, last_epoch)
        print(f"Learning rate scaled by x{self.k}.")
        
    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup:
            warmth = (epoch+1)/self.warmup
            lr = warmth * (0.1*self.k)
        elif epoch < 60:
            lr =  0.1*self.k
        elif epoch < 90:
            lr = 0.01*self.k
        elif epoch <= 120:
            lr = 0.001*self.k
        else:
            lr = 0.001*self.k
        return [lr for base_lr in self.base_lrs]
            
def lambda_(epoch):
    if epoch < 60:
        return 0.1
    elif epoch < 90:
        return 0.01
    elif epoch <= 120:
        return 0.001
    else:
        return 0.001

def get_optimizer(model, SCALER=1, lr=1.0, momentum=0.9, weight_decay=5e-4):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_) if SCALER==1 else BatchScaleLR(optimizer, SCALER)
    return optimizer, scheduler
    
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=3, type=int, help="Number of residual blocks per layer.", required=False)
    parser.add_argument('--num_classes', default=100, type=int, help="Number of classes.", required=False)
    parser.add_argument('--no_skip', default=0, type=int, help="Int flag to disable residual connections.", required=False)
    args = parser.parse_args()
    
    import pdb
    import graphviz
    from torchview import draw_graph
    
    model = ResNet_cifar({'n':args.n,'num_classes':args.num_classes,'no_skip':bool(args.no_skip)})
    graph = draw_graph(model, input_size=(2,3,32,32), device='cpu')
    graph.visual_graph.render()

    print(f"total params:{sum(p.numel() for p in model.parameters())}")
