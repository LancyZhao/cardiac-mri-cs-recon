"""U-Net。重建与分割共用同一个骨干，只换输出层与损失。

刻意保持成"经典 U-Net"：申报书第 275 行批评的正是它上下采样丢细节的问题，
这里把它当 baseline 用，正好对得上项目里的定位。
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, cin: int = 1, cout: int = 1, base: int = 32, depth: int = 4,
                 residual: bool = False):
        """residual=True 时输出 input+delta（重建任务用：网络只学欠采样伪影的残差）。"""
        super().__init__()
        self.residual = residual
        chs = [base * 2 ** i for i in range(depth)]

        self.downs = nn.ModuleList()
        c = cin
        for ch in chs:
            self.downs.append(DoubleConv(c, ch))
            c = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[-1], chs[-1] * 2)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        c = chs[-1] * 2
        for ch in reversed(chs):
            self.ups.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.up_convs.append(DoubleConv(ch * 2, ch))
            c = ch
        self.head = nn.Conv2d(chs[0], cout, 1)

    def forward(self, x):
        inp = x
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = conv(torch.cat([skip, x], dim=1))
        out = self.head(x)
        if self.residual:
            return inp + out
        return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
