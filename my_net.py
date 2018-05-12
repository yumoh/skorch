
from skorch.torch_env import *


class Net(nn.Module):
    """
    work on sklearn.datasets.load_digist shape as (8,8)
    """
    def __init__(self):
        super().__init__()
        # bx1x8x8->bx8x6x6
        self.layer1 = nn.Sequential(nn.BatchNorm2d(1),
                                    nn.Conv2d(1, 8, kernel_size=3),
                                    nn.ReLU())
        # ->bx16x2x2
        self.layer2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))
        # ->24x6x6->30x2x2
        self.layer4 = nn.Sequential(nn.BatchNorm2d(24),
                                    nn.Conv2d(24, 30, kernel_size=3),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.ReLU())

        self.end_layer = nn.Sequential(nn.Conv2d(46, 10, kernel_size=2))

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        up_layer2 = F.upsample(layer2, size=(6, 6))
        # size:bx24x6x6
        layer3 = torch.cat([up_layer2, layer1], dim=1)
        layer4 = self.layer4(layer3)
        layer5 = torch.cat([layer2, layer4], dim=1)
        y = self.end_layer(layer5)
        y = torch.squeeze(y)
        y = F.log_softmax(y, dim=1)
        return y
