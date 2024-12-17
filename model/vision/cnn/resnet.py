import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self , inf , outf ,stride=1 , downsample=None):
        
        super(ResidualBlock , self).__init__()

        self.conv1 = nn.Conv2d(inf , outf , kernel_size=3 , stride=stride , padding=1 , bias=False)
        self.bn2 = nn.BatchNorm2d(outf)
        self.conv2 = nn.Conv2d(inf , outf , kernel_size=3 , stride=stride , padding=1 ,bias=False)
        self.bn2 = nn.BatchNorm2d(outf)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample


    def forward(self ,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity
        out = self.relu(out)

        return out
    


class BottleneckBlock(nn.Module):

    def __init__(self , inf , outf,  stride=1 , downsample=None , expansion=1):

        super(BottleneckBlock , self).__init__()

        self.expansion = expansion
        self.conv1 = nn.Conv2d(inf , outf , kernel_size=1 , bias=False)
        self.bn1 = nn.BatchNorm2d(outf)
        self.conv2 = nn.Conv2d(inf , outf , kernel_size=1 , bias=False)
        self.bn2 = nn.BatchNorm2d(outf)
        self.conv3 = nn.Conv2d(inf , outf * expansion, kernel_size=1 , bias=False)
        self.bn3 = nn.BatchNorm2d(outf * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self , x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)


        out =self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out
    

class ResNet(nn.Module):

    def __init__(self , block , layers , n_class):

        super(ResNet , self).__init__()
        self.in_channels = 64

        # initial layer
        self.conv1 = nn.Conv2d(3 , 64 , kernel_size=7 , stride=2 , padding=3 , bias=False)
        self.bn1  = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1)

        self.layer1 = self._make_layer(block , 64 , layers[0])
        self.layer2 = self._maker_layer(block , 128 , layers[1] , stride=2)
        self.layer3 = self._maker_layer(block  , 256 , layers[2] , stride=2)
        self.layer4 = self._maker_layer(block , 512 , layers[3] , stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion , n_class)

    def _maker_layer(self , block , outf , blocks , stride=1):
        downsample = None

        if stride != 1 or self.in_channels != outf * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels , outf * block.expansion , kernel_size=1 , stride=stride , bias=False),
                nn.BatchNorm2d(outf*block.expansion),
            )

        layers = []

        layers.append(block(self.in_channels , outf , stride , downsample))
        self.in_channels = outf * block.expansion

        for _ in range(1 , blocks):
            layers.append(block(self.in_channels , outf))

        return nn.Sequential(*layers)
    


    def forward(self , x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.avgpool(x)
        x = torch.flatten(x , 1)
        x = self.fc(x)
        

