from __future__ import annotations

import torch
import torch.nn as nn


def group_norm(channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: tuple[int, int, int] = (1, 1, 1), norm_type: str = "gn"):
        super().__init__()
        padding = tuple(dilation)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=padding, dilation=dilation)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        if norm_type == "gn":
            self.n1 = group_norm(out_ch)
            self.n2 = group_norm(out_ch)
        elif norm_type == "none":
            self.n1 = nn.Identity()
            self.n2 = nn.Identity()
        else:
            raise ValueError("norm_type must be 'gn' or 'none'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.n1(self.conv1(x)))
        x = self.act(self.n2(self.conv2(x)))
        return x


class GazeMorph(nn.Module):
    def __init__(self, in_ch: int = 2, base: int = 16, max_disp: float = 24.0):
        super().__init__()
        self.max_disp = max_disp
        self.e1 = ConvBlock3D(in_ch, base, norm_type="gn")
        self.d1 = nn.Conv3d(base, base, 3, stride=(2, 2, 2), padding=1)
        self.e2 = ConvBlock3D(base, base * 2, norm_type="gn")
        self.d2 = nn.Conv3d(base * 2, base * 2, 3, stride=(2, 2, 2), padding=1)
        self.e3 = ConvBlock3D(base * 2, base * 4, norm_type="gn")
        self.d3 = nn.Conv3d(base * 4, base * 4, 3, stride=(2, 2, 2), padding=1)
        self.e4 = ConvBlock3D(base * 4, base * 8, norm_type="gn")
        self.d4 = nn.Conv3d(base * 8, base * 8, 3, stride=(2, 2, 2), padding=1)
        self.b1 = ConvBlock3D(base * 8, base * 16, norm_type="gn")
        self.b2 = ConvBlock3D(base * 16, base * 16, norm_type="gn")
        self.u4 = nn.ConvTranspose3d(base * 16, base * 8, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.dec4 = ConvBlock3D(base * 16, base * 8, norm_type="gn")
        self.u3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base * 8, base * 4, norm_type="gn")
        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2, norm_type="none")
        self.u1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base, norm_type="none")
        self.head = nn.Conv3d(base, 2, kernel_size=3, padding=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous(memory_format=torch.channels_last_3d)
        e1 = self.e1(x)
        e2 = self.e2(self.d1(e1))
        e3 = self.e3(self.d2(e2))
        e4 = self.e4(self.d3(e3))
        b = self.b2(self.b1(self.d4(e4)))
        d4 = self.dec4(torch.cat([self.u4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.u3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.u1(d2), e1], dim=1))
        return torch.tanh(self.head(d1)) * self.max_disp
