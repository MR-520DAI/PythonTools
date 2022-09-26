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
        self.conv1 = nn.Conv2d(3, 24, kernel_size=(3,3), stride=(2,2), padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.layer1 = nn.Sequential(RestNetBasicBlock(24, 24, 1),
                                    RestNetBasicBlock(24, 24, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(24, 32, [2, 1]),
                                    RestNetBasicBlock(32, 32, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(32, 48, [2, 1]),
                                    RestNetBasicBlock(48, 48, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(48, 64, [2, 1]),
                                    RestNetBasicBlock(64, 64, 1))

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x2, x3, x4

class LaneDect(nn.Module):
    def __init__(self):
        super(LaneDect, self).__init__()
        # op
        self.relu = nn.ReLU()
        # backbone
        self.backbone = RestNet18()
        # FPN
        self.FPN4x1 = conv_bn_relu(64, 64, 3, 1, 1)
        self.FPN4x2 = conv_bn_relu(64, 48, 1, 1, 0)
        self.Deconv4 = deconv_bn(48, 48, 4, 2, 1)

        self.FPN3x1 = conv_bn_relu(48, 48, 3, 1, 1)
        self.FPN3x2 = conv_bn_relu(48, 32, 1, 1, 0)
        self.Deconv3 = deconv_bn(32, 32, 4, 2, 1)

        # head
        self.hm1x1 = conv_bn_relu(48, 48, 3, 1, 1)
        self.hm1x2 = nn.Conv2d(48, 1, 3, 1, 1)
        self.res1 = conv_bn_relu(32, 32, 3, 1, 1)
        self.res2 = conv_bn_relu(32, 32, 3, 1, 1)
        self.res3 = conv_bn_relu(32, 32, 3, 1, 1)
        self.seg = nn.Conv2d(32, 4, 3, 1, 1)

        self.range1 = conv_bn_relu(32, 4, 3, 1, 1)
        self.range2 = nn.Conv1d(100, 64, 1, 1)
        self.range3 = nn.Conv1d(64, 2, 1, 1)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        p2, p3, p4 = self.backbone(x)   #p2(n,32,40,100) p3(n,48,20,50) p4(n,64,10,25)

        p4 = self.FPN4x1(p4)
        p4 = self.FPN4x2(p4)
        p4 = self.Deconv4(p4)   # p4(n, 48, 20, 50)

        p3 = self.relu(p3 + p4)

        hm = self.hm1x1(p3)
        hm = self.hm1x2(hm)
        hm = self.sigmoid(hm)

        p3 = self.FPN3x1(p3)
        p3 = self.FPN3x2(p3)
        p3 = self.Deconv3(p3)   # p3(n, 32, 40, 100)

        p2 = self.relu(p2 + p3) # p2(n, 32, 40, 100)

        res1 = self.res1(p2)
        res1 = self.relu(p2 + res1) # res1(n, 32, 40, 100)

        res2 = self.res2(res1)
        res2 = self.relu(res1 + res2)   # res2(n, 32, 40, 100)

        res3 = self.res3(res2)
        res3 = self.relu(res2 + res3)   # res3(n, 32, 40, 100)

        pre_seg = self.seg(res3)

        pre_range = self.range1(res3)
        pre_range = pre_range.permute(0, 1, 3, 2).reshape((pre_range.shape[0]*pre_range.shape[1]), pre_range.shape[3], pre_range.shape[2])
        pre_range = self.range2(pre_range)
        pre_range = F.relu(pre_range)
        pre_range = self.range3(pre_range)
        pre_range = pre_range.reshape(res3.shape[0], 4, 2, 40).permute(0, 2, 1, 3)
        return pre_seg, pre_range, hm

if __name__ == '__main__':

    net = LaneDect()
    net.eval()

    input = torch.randn(2, 3, 320, 800, device='cpu')
    out1, out2, out3 = net(input)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    print(1111)
    torch.save(net.state_dict(), "./LaneDet_0.pth")