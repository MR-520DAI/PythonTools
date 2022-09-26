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

class deconv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(deconv_bn_relu, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=(2,2), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = nn.Sequential(RestNetBasicBlock(32, 32, 1),
                                    RestNetBasicBlock(32, 32, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(32, 48, [2, 1]),
                                    RestNetBasicBlock(48, 48, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(48, 64, [2, 1]),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(64, 96, [2, 1]),
                                    RestNetBasicBlock(96, 96, 1))

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4

class LaneDect(nn.Module):
    def __init__(self):
        super(LaneDect, self).__init__()
        # op
        self.relu = nn.ReLU()
        # backbone
        self.backbone = RestNet18()
        # FPN
        self.FPN4x1 = conv_bn_relu(96, 64, 1, 1, 0)
        self.FPN3x1 = conv_bn_relu(64, 64, 1, 1, 0)
        self.FPN2x1 = conv_bn_relu(48, 64, 1, 1, 0)
        self.FPN1x1 = conv_bn_relu(32, 64, 1, 1, 0)

        # head
        self.res1 = conv_bn_relu(64, 64, 3, 1, 1)
        self.res2 = conv_bn_relu(64, 64, 3, 1, 1)
        self.res3 = conv_bn_relu(64, 64, 3, 1, 1)

        self.seg1 = conv_bn_relu(64, 64, 3, 1, 1)
        self.seg2 = conv_bn_relu(64, 64, 3, 1, 1)
        self.seg3 = torch.nn.Conv2d(64, 4, 3, 1, 1)

        self.offset1 = conv_bn_relu(64, 64, 3, 1, 1)
        self.offset2 = conv_bn_relu(64, 64, 3, 1, 1)
        self.offset3 = torch.nn.Conv2d(64, 1, 3, 1, 1)

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        p1, p2, p3, p4 = self.backbone(x)   #p1(n,24,80,200) p2(n,32,40,100) p3(n,48,20,50) p4(n,64,10,25)

        p4 = self.FPN4x1(p4)
        p4 = F.interpolate(p4, scale_factor=2, mode='bilinear')
        
        p3 = self.FPN3x1(p3)
        p3 = p3 + p4

        p3 = F.interpolate(p3, scale_factor=2, mode='bilinear')
        p2 = self.FPN2x1(p2)
        p2 = p2 + p3

        p2 = F.interpolate(p2, scale_factor=2, mode='bilinear')
        p1 = self.FPN1x1(p1)
        p1 = p1 + p2

        res1 = self.res1(p1)
        res1 = res1 + p1
        res2 = self.res2(res1)
        res2 = res1 + res2
        res3 = self.res3(res2)
        res3 = res2 + res3

        seg = self.seg1(res3)
        seg = self.seg2(seg)
        seg = self.seg3(seg)
        seg = self.softmax(seg)
        # seg = seg.permute(0, 3, 1, 2)

        offset = self.offset1(res3)
        offset = self.offset2(offset)
        offset = self.offset3(offset)
        offset = self.sigmoid(offset)

        return seg, offset

if __name__ == '__main__':

    net = LaneDect()
    net.eval()

    input = torch.randn(2, 3, 320, 800, device='cpu')
    out1, out2 = net(input)
    print(out1.shape)
    print(out2.shape)
    print(1111)
    torch.save(net.state_dict(), "./LaneDet_0.pth")