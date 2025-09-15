import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.layernorm import LayerNorm2d

def moment(x, dim=(2, 3), k=2):
    assert len(x.size()) == 4
    mean = torch.mean(x, dim=dim).unsqueeze(-1).unsqueeze(-1)
    mk = (1 / (x.size(2) * x.size(3))) * torch.sum(torch.pow(x - mean, k), dim=dim)
    return mk



class clffm(nn.Module):
    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
        super(clffm, self).__init__()
        f = esa_channels

        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)

        self.conv2_3x3 = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1, groups=f),
            nn.Conv2d(f, f, kernel_size=1)
        )

        self.conv2_5x5 = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=5, stride=2, padding=2, groups=f),
            nn.Conv2d(f, f, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=3, padding=1, groups=f),
            nn.Conv2d(f, f, kernel_size=1)
        )
        self.conv4 = conv(f, n_feats, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c2_3x3 = self.conv2_3x3(c1_)
        c2_5x5 = self.conv2_5x5(c1_)
        c2 = c2_3x3 + c2_5x5
        c3 = self.conv3(c2)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m
