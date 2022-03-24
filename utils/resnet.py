import torch.nn as nn

# 残差块
class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample # downsample 对输入特征图大小进行减半处理
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)
            
    # 残差块里首先有 2 个相同输出通道数的 卷积层,
    # 每个卷积层后接一个批量归一化层和 ReLU 激活函数*,
    # 然后我们将输入跳过这 2 个卷积运算后直接加在最后的 ReLU 激活函数*前.
    # *此文件中是 PReLU 激活函数
    # PReLU 和 ReLU 的区别主要是前者在 输入小于 0 的部分加了一个系数 a,
    # 若 a ==0, PReLU 退化为 ReLU；若 a 很小(比如0.01) ,PReLU 退化为 LReLU,
    # 有实验证明，与ReLU相比，LReLU对最终的结果几乎没什么影响。
    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNet(nn.Module):
    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNet, self).__init__()
        # 所有 ResNet 网络的输入由一个大卷积核+最大池化组成，极大减少了存储所需大小
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()
        self.fc5 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1d(512)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)
        layers = [block(self.inplanes, planes, stride, downsample, use_se=self.use_se)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))

        return nn.Sequential(*layers)
    
    # 规定了网络数据的流向
    def forward(self, x):
        # 输入
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)
        # 中间卷积
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 输出
        x = self.pool(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc5(x)
        x = self.bn5(x)

        return x

# 3,4,6,3 是 ResNet34 卷积部分的配置，至于为什么要用这个配置没有解释清楚
def resnet34(use_se=True):
    model = ResNet(IRBlock, [3, 4, 6, 3], use_se=use_se)
    return model
