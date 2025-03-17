import torch
import torch.nn as nn
import torch.nn.functional as F

# Создание модели UNet
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Энкодер
        self.enc1 = DoubleConv(1, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        # Узкое место
        self.bottleneck = DoubleConv(64, 128)
        # Декодер
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = DoubleConv(128 + 64, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DoubleConv(64 + 32, 32)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DoubleConv(32 + 16, 16)

        # Дополнительные свёрточные слои для расчёта A
        self.A_conv = nn.Conv2d(16, 24, kernel_size=3, padding=1)

        # Итоговой свёрточный слой
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Энкодер
        x1 = self.enc1(x)  # 256x256x16
        p1 = self.pool1(x1)  # 128x128x16
        x2 = self.enc2(p1)  # 128x128x32
        p2 = self.pool2(x2)  # 64x64x32
        x3 = self.enc3(p2)  # 64x64x64
        p3 = self.pool3(x3)  # 32x32x64
        # Узкое место
        b = self.bottleneck(p3)  # 32x32x128
        # Декодер
        u3 = self.up3(b)
        cat3 = torch.cat([u3, x3], dim=1)
        d3 = self.dec3(cat3)
        u2 = self.up2(d3)
        cat2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(cat2)
        u1 = self.up1(d2)
        cat1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(cat1)

        # Вычисляем A
        A = torch.tanh(self.A_conv(d1))  # Аналогично x_r из enhance_net_nopool
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(A, 3, dim=1)

        # Генерируем итоговое изображение (аналогичное enhance_net_nopool)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)

        return enhance_image_1, enhance_image, A