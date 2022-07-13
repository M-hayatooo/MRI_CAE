from tkinter import N

import torch
# from torch import Module
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.avgpool = nn.AvgPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(10 * 12 * 10 * 64, 512)

        self.sigmoid = nn.Sigmoid()
        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(3)
        self.batchnorm3d3 = nn.BatchNorm3d(32)
        self.batchnorm3d4 = nn.BatchNorm3d(64)

    #    self.batchnorm1 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = F.relu(self.batchnorm3d1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm3d2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm3d3(self.conv3(x)))
        x = F.relu(self.batchnorm3d4(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, 10 * 12 * 10 * 64)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.batchnorm1(self.fc1(x))) ここの batchnorm は要らない
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsamp1 = nn.Upsample((20, 24, 20), mode='nearest')  # mode="trilinear", align_corners=True
        self.upsamp2 = nn.Upsample((40, 48, 40), mode='nearest')  # mode="nearest"  degault mode == nearest
        self.upsamp3 = nn.Upsample((80, 96, 80), mode='nearest')
      # self.upsamp3 = nn.Upsample((80, 96, 80), mode='nearest', align_corners=False)

        self.dfc1 = nn.Linear(512, 10 * 12 * 10 * 64)
        self.deconv1 = nn.ConvTranspose3d(64, 32, 3, padding=1)
        self.deconv2 = nn.ConvTranspose3d(32, 3, 3, padding=1)
        self.deconv3 = nn.ConvTranspose3d(3, 3, 3, padding=1)
        self.deconv4 = nn.ConvTranspose3d(3, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.batchnorm_d1 = nn.BatchNorm1d(76800)
        self.batchnorm_d3d1 = nn.BatchNorm3d(32)
        self.batchnorm_d3d2 = nn.BatchNorm3d(3)
        self.batchnorm_d3d3 = nn.BatchNorm3d(3)

    def forward(self, x):
        x = F.relu(self.batchnorm_d1(self.dfc1(x)))
        x = x.view(-1, 64, 10, 12, 10)
        x = self.upsamp1(x)
        x = F.relu(self.batchnorm_d3d1(self.deconv1(x)))
        x = F.relu(self.batchnorm_d3d2(self.deconv2(x)))
        x = self.upsamp2(x)
        x = F.relu(self.batchnorm_d3d3(self.deconv3(x)))
        x = self.upsamp3(x)
        x = self.sigmoid(self.deconv4(x))
        return x


class LuckyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 32, 3, padding=1)
        self.fc1 = nn.Linear(10 * 12 * 10 * 32, 512)
        self.fc2 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(p=0.25)
        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(3)
        self.batchnorm3d3 = nn.BatchNorm3d(32)
        self.batchnorm3d4 = nn.BatchNorm3d(32)
        self.batchnorm3d5 = nn.BatchNorm3d(64)
        self.batchnorm3d6 = nn.BatchNorm3d(64)
        self.batchnorm1d = nn.BatchNorm1d(512)

    def forward(self, x):  #  80 × 96 × 80 = 614,400
        x = self.pool(
            x
        )  #                             #                          {1 × 80 × 96 × 80} ⇒ {1 × 40 × 48 × 40}
        x = F.relu(
            self.batchnorm3d1(self.conv1(x))
        )  # 1channel → 3channels  ## {1 × 40 × 48 × 40} ⇒ {3 × 40 × 48 × 40}
        x = F.relu(
            self.batchnorm3d2(self.conv2(x))
        )  # 3channels→ 3channels  ## {3 × 40 × 48 × 40} ⇒ {3 × 40 × 48 × 40}
        x = self.pool(x)
        x = F.relu(
            self.batchnorm3d3(self.conv3(x))
        )  #                       ## {3 × 20 × 24 × 20} ⇒{32 × 20 × 24 × 20}
        x = F.relu(
            self.batchnorm3d4(self.conv4(x))
        )  #                       ## {32 × 20 × 24 × 20} ⇒{32 × 20 × 24 × 20}
        x = self.pool(x)  #                             #  64 × 10 × 12 × 10
        x = x.view(-1, 10 * 12 * 10 * 32)  #     10 * 12 * 10 * 32 = 38,400
        x = self.dropout(x)
        x = F.relu(
            self.batchnorm1d(self.fc1(x))
        )  #        # {32 × 10 × 12 × 10 = 38,400} ⇒ {512}
        x = self.dropout(x)
        x = self.fc2(x)  #          { 512 } ⇒ { 2 } 2クラス分類
        return x


class LuckyNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 32, 3, padding=1)
        self.fc1 = nn.Linear(10 * 12 * 10 * 32, 512)
        self.fc2 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(p=0.25)
        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(3)
        self.batchnorm3d3 = nn.BatchNorm3d(32)
        self.batchnorm3d4 = nn.BatchNorm3d(32)
        self.batchnorm3d5 = nn.BatchNorm3d(32)
        self.batchnorm3d6 = nn.BatchNorm3d(32)
        # self.batchnorm1d = nn.BatchNorm1d(512)

    # 畳み込み⇒畳み込み⇒poolだとあんま安定しなかった
    def forward(self, x):  #  80 × 96 × 80 = 614,400
        x = self.pool(
            x
        )  #                             #                          {1 × 80 × 96 × 80} ⇒ {1 × 40 × 48 × 40}
        x = F.relu(
            self.batchnorm3d1(self.conv1(x))
        )  # 1channel → 3channels  ## {1 × 40 × 48 × 40} ⇒ {3 × 40 × 48 × 40}
        x = F.relu(
            self.batchnorm3d2(self.conv2(x))
        )  # 3channels→ 3channels  ## {3 × 40 × 48 × 40} ⇒ {3 × 40 × 48 × 40}
        x = self.pool(x)
        x = F.relu(
            self.batchnorm3d3(self.conv3(x))
        )  # 1channel → 3channels  ## {3 × 40 × 48 × 40} ⇒ {32 × 40 × 48 × 40}
        x = F.relu(
            self.batchnorm3d4(self.conv4(x))
        )  # 3channels→ 3channels  ## {3 × 40 × 48 × 40} ⇒ {3 × 40 × 48 × 40}
        x = self.pool(x)
        x = x.view(-1, 10 * 12 * 10 * 32)  #     10 * 12 * 10 * 32 = 38,400
        x = self.dropout(x)
        x = F.relu(self.fc1(x))  #        # {32 × 10 × 12 × 10 = 38,400} ⇒ {512}
        x = self.dropout(x)
        x = self.fc2(x)  #          { 512 } ⇒ { 2 } 2クラス分類
        return x


class LuckyNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(10 * 12 * 10 * 64, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 2)

        # self.dropout = nn.Dropout(p=0.2)
        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(3)
        self.batchnorm3d3 = nn.BatchNorm3d(32)
        self.batchnorm3d4 = nn.BatchNorm3d(64)
        self.batchnorm3d5 = nn.BatchNorm3d(32)
        self.batchnorm3d6 = nn.BatchNorm3d(32)
        # self.batchnorm1d = nn.BatchNorm1d(512)

    def forward(self, x):  #  80 × 96 × 80 = 614,400
        x = self.pool(
            x
        )  #                             #                          {1 × 80 × 96 × 80} ⇒ {1 × 40 × 48 × 40}
        x = F.relu(
            self.batchnorm3d1(self.conv1(x))
        )  # 1channel → 3channels  ## {1 × 40 × 48 × 40} ⇒ {3 × 40 × 48 × 40}
        x = F.relu(
            self.batchnorm3d2(self.conv2(x))
        )  # 3channels→ 3channels  ## {3 × 40 × 48 × 40} ⇒ {3 × 40 × 48 × 40}
        x = self.pool(x)
        x = F.relu(
            self.batchnorm3d3(self.conv3(x))
        )  #                       ## {3 × 20 × 24 × 20} ⇒{32 × 20 × 24 × 20}
        x = F.relu(
            self.batchnorm3d4(self.conv4(x))
        )  #                       ## {32 × 20 × 24 × 20} ⇒{32 × 20 × 24 × 20}
        x = self.pool(x)
        x = x.view(
            -1, 10 * 12 * 10 * 32
        )  #                                       10 * 12 * 10 * 32 = 38,400
        x = F.relu(
            self.fc1(x)
        )  #                                      #  {32 × 10 × 12 × 10 = 38,400} ⇒ {512}
        x = F.relu(
            self.fc2(x)
        )  #                                      #  {32 × 10 × 12 × 10 = 38,400} ⇒ {512}
        x = self.fc3(x)  #          { 512 } ⇒ { 2 } 2クラス分類
        return x


class LuckyNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv4 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv5 = nn.Conv3d(32, 32, 3, padding=1)
        self.fc1 = nn.Linear(10 * 12 * 10 * 32, 512)
        self.fc2 = nn.Linear(512, 2)

        # self.dropout = nn.Dropout(p=0.2)
        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(3)
        self.batchnorm3d3 = nn.BatchNorm3d(32)
        self.batchnorm3d4 = nn.BatchNorm3d(32)
        self.batchnorm3d5 = nn.BatchNorm3d(64)
        self.batchnorm3d6 = nn.BatchNorm3d(64)
        self.batchnorm1d = nn.BatchNorm1d(512)

    def forward(self, x):  #  80 × 96 × 80 = 614,400
        x = self.pool(
            x
        )  #                             #                               {1 × 80 × 96 × 80} ⇒ {1 × 40 × 48 × 40}
        x1 = F.relu(
            self.batchnorm3d1(self.conv1(x))
        )  #     1channel → 3channels  ## {1 × 40 × 48 × 40} ⇒ {3 × 40 × 48 × 40}
        x = F.relu(
            self.batchnorm3d2(self.conv2(x + x1))
        )  # 3channels→ 3channels  ## {3 × 40 × 48 × 40} ⇒ {3 × 40 × 48 × 40}
        x = self.pool(
            x
        )  #                             #                            ## {3 × 40 × 48 × 40} ⇒ {3 × 20 × 24 × 20}
        x1 = F.relu(
            self.batchnorm3d3(self.conv3(x))
        )  #                           ## {3 × 20 × 24 × 20} ⇒{32 × 20 × 24 × 20}
        x = F.relu(
            self.batchnorm3d4(self.conv4(x + x1))
        )  #                       ## {32 × 20 × 24 × 20} ⇒{32 × 20 × 24 × 20}
        x = self.pool(
            x
        )  #                                                          #  64 × 10 × 12 × 10
        x = x.view(
            -1, 10 * 12 * 10 * 32
        )  #                                           10 * 12 * 10 * 32 = 38,400
        # x = self.dropout(x)
        x = F.relu(
            self.batchnorm1d(self.fc1(x))
        )  #                                      #  {32 × 10 × 12 × 10 = 38,400} ⇒ {512}
        # x = self.dropout(x)
        x = self.fc2(x)  #          { 512 } ⇒ { 2 } 2クラス分類
        return x


class LuckyNet4(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv3d(32, 32, 3, padding=1)
        self.fc1 = nn.Linear(10 * 12 * 10 * 32, 512)
        self.fc2 = nn.Linear(512, 2)

        # self.dropout = nn.Dropout(p=0.2)
        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(3)
        self.batchnorm3d3 = nn.BatchNorm3d(32)
        self.batchnorm3d4 = nn.BatchNorm3d(32)
        self.batchnorm3d5 = nn.BatchNorm3d(32)
        self.batchnorm3d6 = nn.BatchNorm3d(32)
        self.batchnorm3d7 = nn.BatchNorm3d(64)
        self.batchnorm3d8 = nn.BatchNorm3d(64)
        # self.batchnorm1d = nn.BatchNorm1d(512)

    # 畳み込み⇒畳み込み⇒poolだとあんま安定しなかった
    def forward(self, x):  #  80 × 96 × 80 = 614,400
        x = self.pool(x)  #                             #                          {1 × 80 × 96 × 80} ⇒ {1 × 40 × 48 × 40}
        x = F.relu(self.batchnorm3d1(self.conv1(x)))  # 1channel → 3channels  ## {1 × 40 × 48 × 40} ⇒ {3 × 40 × 48 × 40}
        x = F.relu(self.batchnorm3d2(self.conv2(x)))  # 3channels→ 3channels  ## {3 × 40 × 48 × 40} ⇒ {3 × 40 × 48 × 40}
        x = self.pool(x)
        x = F.relu(self.batchnorm3d3(self.conv3(x)))  # 1channel → 3channels  ## {3 × 40 × 48 × 40} ⇒ {32 × 40 × 48 × 40}
        x = F.relu(self.batchnorm3d4(self.conv4(x)))  # 3channels→ 3channels  ## {3 × 40 × 48 × 40} ⇒ {3 × 40 × 48 × 40}
        x = self.pool(x)
        x = F.relu(self.batchnorm3d5(self.conv5(x)))  #                       ## {3 × 20 × 24 × 20} ⇒{32 × 20 × 24 × 20}
        x = F.relu(self.batchnorm3d6(self.conv6(x)))  #                       ## {32 × 20 × 24 × 20} ⇒{32 × 20 × 24 × 20}
        x = self.pool(x)  #                             #  64 × 10 × 12 × 10
        x = x.view(-1, 10 * 12 * 10 * 32)  #     10 * 12 * 10 * 32 = 38,400
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))  #        # {32 × 10 × 12 × 10 = 38,400} ⇒ {512}
        # x = self.dropout(x)
        x = self.fc2(x)  #          { 512 } ⇒ { 2 } 2クラス分類
        return x


class FujiNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(10 * 12 * 10 * 64, 512)
        self.fc2 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d12 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(32)
        self.batchnorm3d3 = nn.BatchNorm3d(64)

    #    self.batchnorm1 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.pool(x)
        x = F.relu(self.batchnorm3d1(self.conv1(x)))
        x = F.relu(self.batchnorm3d12(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm3d2(self.conv3(x)))
        x = F.relu(self.batchnorm3d3(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, 10 * 12 * 10 * 64)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FujiNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(10 * 12 * 10 * 64, 512)
        self.fc2 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(p=0.25)
        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d12 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(32)
        self.batchnorm3d3 = nn.BatchNorm3d(64)

    #    self.batchnorm1 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.pool(x)
        x = F.relu(self.batchnorm3d1(self.conv1(x)))
        x = F.relu(self.batchnorm3d12(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm3d2(self.conv3(x)))
        x = F.relu(self.batchnorm3d3(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, 10 * 12 * 10 * 64)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FujiNet_avg(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(10 * 12 * 10 * 64, 512)
        self.fc2 = nn.Linear(512, 2)

        # self.dropout = nn.Dropout(p=0.25)
        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d12 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(32)
        self.batchnorm3d3 = nn.BatchNorm3d(64)

    #    self.batchnorm1 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.pool(x)
        x = F.relu(self.batchnorm3d1(self.conv1(x)))
        x = F.relu(self.batchnorm3d12(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm3d2(self.conv3(x)))
        x = F.relu(self.batchnorm3d3(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, 10 * 12 * 10 * 64)
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x


class Cae(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def __call__(self, x):
        x = self.forward(x)
        return x


class Vae(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc_mu = nn.Linear(512, 150)
        self.fc_logvar = nn.Linear(512, 150)
        self.dfc2 = nn.Linear(150, 512)

    def get_mu_var(self, x):
        x = self.encoder(x)
        # mu = self.encoder.fc_mu(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparamenterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def encode(self, x):
        mu, logvar = self.get_mu_var(x)
        return self.reparamenterize(mu, logvar)

    def decode(self, x):
        x = F.relu(self.dfc2(x))
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.get_mu_var(x)
        z = self.reparamenterize(mu, logvar)
        x_re = self.decode(z)
        return x_re, mu, logvar

    def loss(self, x_re, x, mu, logvar):
        re_err = F.binary_cross_entropy(x_re, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return re_err + kld


# ----------------------------------------------------------------------------------------------------------- #


class Vaee(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(10 * 12 * 10 * 64, 512)
        self.fc_mu = nn.Linear(512, 150)
        self.fc_logvar = nn.Linear(512, 150)

        # self.unpool = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.upsamp1 = nn.Upsample((20, 24, 20), mode="trilinear", align_corners=True)
        self.upsamp2 = nn.Upsample((40, 48, 40), mode="trilinear", align_corners=True)
        self.upsamp3 = nn.Upsample((80, 96, 80), mode="trilinear", align_corners=True)
        self.dfc2 = nn.Linear(150, 512)
        self.dfc1 = nn.Linear(512, 10 * 12 * 10 * 64)
        self.deconv1 = nn.ConvTranspose3d(64, 32, 3, padding=1)
        self.deconv2 = nn.ConvTranspose3d(32, 3, 3, padding=1)
        self.deconv3 = nn.ConvTranspose3d(3, 3, 3, padding=1)
        self.deconv4 = nn.ConvTranspose3d(3, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d12 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(32)
        self.batchnorm3d3 = nn.BatchNorm3d(64)
        self.batchnorm1 = nn.BatchNorm1d(512)

        self.batchnorm_d3d1 = nn.BatchNorm3d(32)
        self.batchnorm_d3d2 = nn.BatchNorm3d(3)
        self.batchnorm_d3d3 = nn.BatchNorm3d(3)
        self.batchnorm_d1 = nn.BatchNorm1d(64000)

    def get_mu_var(self, x):
        x = F.relu(self.batchnorm3d1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm3d12(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm3d2(self.conv3(x)))
        x = F.relu(self.batchnorm3d3(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, 10 * 12 * 10 * 64)
        x = F.relu(self.batchnorm1(self.fc1(x)))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparamenterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def decode(self, x):
        x = F.relu(self.dfc2(x))
        x = F.relu(self.batchnorm_d1(self.dfc1(x)))
        x = x.view(-1, 64, 10, 10, 10)
        x = self.upsamp1(x)
        x = F.relu(self.batchnorm_d3d1(self.deconv1(x)))
        x = F.relu(self.batchnorm_d3d2(self.deconv2(x)))
        x = self.upsamp2(x)
        x = F.relu(self.batchnorm_d3d3(self.deconv3(x)))
        x = self.upsamp3(x)
        x = self.sigmoid(self.deconv4(x))
        return x

    def encode(self, x):
        mu, logvar = self.get_mu_var(x)
        return self.reparamenterize(mu, logvar)

    def forward(self, x):
        mu, logvar = self.get_mu_var(x)
        z = self.reparamenterize(mu, logvar)
        x_re = self.decode(z)
        return x_re, mu, logvar

    def loss(self, x_re, x, mu, logvar):

        # bsize = x.size(0)
        # delta = 1e-8
        # x = x.view(bsize, -1)
        # x_re = x.view(bsize, -1)
        # mu = mu.view(bsize, -1)
        # logvar = logvar.view(bsize, -1)

        # re_err = torch.sum(torch.sum(x * torch.log(x_re + delta) + (1-x) * torch.log(1-x_re+delta), dim=1), dim=0)
        re_err = F.binary_cross_entropy(x_re, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
        return re_err + kld


class CAE3(nn.Module):
    def __init__(self, featureDim=1 * 5 * 5 * 5, zDim=125):
        super().__init__()
        self.KER_N = 27
        self.KETS_N = 1
        # Encoder
        self.conv1 = nn.Conv3d(1, self.KER_N, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool3d(2, 2)
        self.conv2 = nn.Conv3d(self.KER_N, self.KER_N, 3, 1, 1)
        self.pool2 = nn.AvgPool3d(2, 2)
        self.conv3 = nn.Conv3d(self.KER_N, self.KER_N, 3, 1, 1)
        self.conv4 = nn.Conv3d(self.KER_N, self.KER_N, 3, 1, 1)
        self.conv5 = nn.Conv3d(self.KER_N, self.KER_N, 3, 1, 1)
        self.pool3 = nn.AvgPool3d(2, 2)
        self.conv6 = nn.Conv3d(self.KER_N, self.KER_N, 3, 1, 1)
        self.conv7 = nn.Conv3d(self.KER_N, self.KER_N, 3, 1, 1)
        self.conv8_1 = nn.Conv3d(self.KER_N, self.KETS_N, 3, 1, 1)
        self.conv8_2 = nn.Conv3d(self.KER_N, self.KETS_N, 3, 1, 1)
        self.pool4 = nn.AvgPool3d(2, 2)
        # Decoder
        self.unpool4 = nn.Upsample(
            scale_factor=(2, 2, 2), mode="trilinear", align_corners=True
        )
        self.deconv8_2 = nn.Conv3d(self.KETS_N, self.KER_N, 3, 1, 1)
        self.deconv8_1 = nn.Conv3d(self.KETS_N, self.KER_N, 3, 1, 1)
        self.deconv7 = nn.ConvTranspose3d(self.KER_N, self.KER_N, 3, 1, 1)
        self.deconv6 = nn.ConvTranspose3d(self.KER_N, self.KER_N, 3, 1, 1)
        self.unpool3 = nn.Upsample(
            scale_factor=(2, 2, 2), mode="trilinear", align_corners=True
        )
        self.deconv5 = nn.ConvTranspose3d(self.KER_N, self.KER_N, 3, 1, 1)
        self.deconv4 = nn.ConvTranspose3d(self.KER_N, self.KER_N, 3, 1, 1)
        self.deconv3 = nn.ConvTranspose3d(self.KER_N, self.KER_N, 3, 1, 1)
        self.unpool2 = nn.Upsample(
            scale_factor=(2, 2, 2), mode="trilinear", align_corners=True
        )
        self.deconv2 = nn.ConvTranspose3d(self.KER_N, self.KER_N, 3, 1, 1)
        self.unpool1 = nn.Upsample(
            scale_factor=(2, 2, 2), mode="trilinear", align_corners=True
        )
        self.deconv1 = nn.ConvTranspose3d(self.KER_N, 1, 3, 1, 1)
        # function
        self.sigmoid = nn.Sigmoid()

    def en(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        t = self.pool2(x)
        x = F.relu(self.conv3(t))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x) + t)
        t = self.pool3(x)
        x = F.relu(self.conv6(t))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8_1(x)) + F.relu(self.conv8_2(t))
        vec = self.pool4(x)
        return vec

    def de(self, vec):
        t = self.unpool4(vec)
        x = F.relu(self.deconv8_1(t))
        x = F.relu(self.deconv7(x))
        x = F.relu(self.deconv6(x)) + F.relu(self.deconv8_2(t))
        t = self.unpool3(x)
        x = F.relu(self.deconv5(t))
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv3(x) + t)
        x = self.unpool2(x)
        x = F.relu(self.deconv2(x))
        x = self.unpool1(x)
        x = self.deconv1(x)
        out = self.sigmoid(x)
        return out

    def forward(self, x):
        vec = self.en(x)
        out = self.de(vec)
        return out, vec

    def __call__(self, x):
        x, _ = self.forward(x)
        return x


class Caee(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(10 * 12 * 10 * 64, 512)
        self.en_norm3d1 = nn.BatchNorm3d(3)
        self.en_norm3d2 = nn.BatchNorm3d(3)
        self.en_norm3d3 = nn.BatchNorm3d(32)
        self.en_norm3d4 = nn.BatchNorm3d(64)

        # self.unpool = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.upsamp1 = nn.Upsample((20, 24, 20), mode="trilinear", align_corners=True)
        self.upsamp2 = nn.Upsample((40, 48, 40), mode="trilinear", align_corners=True)
        self.upsamp3 = nn.Upsample((80, 96, 80), mode="trilinear", align_corners=True)
        self.dfc1 = nn.Linear(512, 10 * 12 * 10 * 64)
        self.deconv1 = nn.ConvTranspose3d(64, 32, 3, padding=1)
        self.deconv2 = nn.ConvTranspose3d(32, 3, 3, padding=1)
        self.deconv3 = nn.ConvTranspose3d(3, 3, 3, padding=1)
        self.deconv4 = nn.ConvTranspose3d(3, 1, 3, padding=1)
        self.de_norm1d1 = nn.BatchNorm1d(10 * 12 * 10 * 64)
        self.de_norm3d2 = nn.BatchNorm3d(32)
        self.de_norm3d3 = nn.BatchNorm3d(3)
        self.de_norm3d4 = nn.BatchNorm3d(3)
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        x = F.relu(self.en_norm3d1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.en_norm3d2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.en_norm3d3(self.conv3(x)))
        x = F.relu(self.en_norm3d4(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, 10 * 12 * 10 * 64)
        x = F.relu(self.fc1(x))
        return x

    def decoder(self, x):
        x = F.relu(self.de_norm1d1(self.dfc1(x)))
        x = x.view(-1, 64, 10, 12, 10)
        x = self.upsamp1(x)
        x = F.relu(self.de_norm3d2(self.deconv1(x)))
        x = F.relu(self.de_norm3d3(self.deconv2(x)))
        x = self.upsamp2(x)
        x = F.relu(self.de_norm3d4(self.deconv3(x)))
        x = self.upsamp3(x)
        x = self.sigmoid(self.deconv4(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def __call__(self, x):
        x = self.forward(x)
        return x


class MyVgg16(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv3d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv3d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv3d(256, 256, 3, padding=1)
        self.conv8 = nn.Conv3d(256, 512, 3, padding=1)
        self.conv9 = nn.Conv3d(512, 512, 3, padding=1)
        self.conv10 = nn.Conv3d(512, 512, 3, padding=1)
        self.conv11 = nn.Conv3d(512, 512, 3, padding=1)
        self.conv12 = nn.Conv3d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv3d(512, 512, 3, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(2 * 2 * 2 * 512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # 80, 80, 80
        x = self.pool(x)  # 40, 40, 40
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)  # 20, 20, 20
        x = self.batchnorm1(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pool(x)  # 10, 10, 10
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pool(x)  # 5, 5, 5
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.pool(x)  # 2, 2, 2
        x = x.view(-1, 2 * 2 * 2 * 512)
        x = F.relu(self.fc1(x))
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
