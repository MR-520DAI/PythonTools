import torch
import torch.nn as nn
from torch.nn import functional as F

class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, stride = stride, padding = padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv_bn(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(conv_bn,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size, stride = stride, padding = padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class deconv_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(deconv_bn, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        return x

class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)

class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)

class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3,3), stride=(2,2), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = nn.Sequential(RestNetBasicBlock(16, 16, 1),
                                    RestNetBasicBlock(16, 16, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(16, 32, [2, 1]),
                                    RestNetBasicBlock(32, 32, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(32, 64, [2, 1]),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4

class LaneDect(nn.Module):
    def __init__(self):
        super(LaneDect, self).__init__()
        # op
        self.relu = nn.ReLU()
        # backbone
        self.backbone = RestNet18()
        # neck
        self.neck1 = conv_bn_relu(128, 64, 3, 1, 1)
        self.neck2 = conv_bn_relu(64, 4, 1, 1, 0)
        # head
        self.head_exit = torch.nn.Sequential(
            torch.nn.Linear(1000, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
        )
        self.soft_max = nn.Softmax(dim=-1)

        self.head_point = torch.nn.Sequential(
            torch.nn.Linear(1000, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1280),
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        backbone = self.backbone(x)   # input(n,128,10,25)

        neck = self.neck1(backbone)
        neck = self.neck2(neck).view(-1, 1000)

        lane_exit = self.head_exit(neck).view(-1, *(32, 2))
        lane_exit = self.soft_max(lane_exit)
        lane_point = self.head_point(neck).view(-1, *(32, 40))
        lane_point = self.sigmoid(lane_point)
        
        return lane_exit, lane_point

if __name__ == '__main__':

    net = LaneDect()
    net.eval()

    input = torch.randn(2, 3, 320, 800, device='cpu')
    out1, out2 = net(input)
    print(out1.shape)
    print(out2.shape)
    print(1111)
    torch.save(net.state_dict(), "./LaneDet_0.pth")