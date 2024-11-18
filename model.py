import torch
import torch.nn as nn
from common_blocks.tcn import TemporalBlock
from options.ADFECG_parameter import parser
from torch.nn import functional as F

class Generator(nn.Module):
    def __init__(self, input_size, Gkernel_size, dropout):
        """
        Generator Network: adopt the framework of Encoder-Decoder
        """
        super(Generator, self).__init__()
        self.Encoder1 = TemporalBlock(n_inputs=input_size, n_outputs=16, kernel_size=Gkernel_size, stride=1, dilation=1,
                                      padding=2, dropout=dropout)
        self.down_sampling1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.Encoder2 = TemporalBlock(n_inputs=16, n_outputs=32, kernel_size=Gkernel_size, stride=1, dilation=2,
                                      padding=4, dropout=dropout)
        self.down_sampling2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.Encoder3 = TemporalBlock(n_inputs=32, n_outputs=64, kernel_size=Gkernel_size, stride=1, dilation=4,
                                      padding=8, dropout=dropout)
        # decoder: TCN 网络实现
        self.up_sampling1 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.Decoder1 = TemporalBlock(n_inputs=64, n_outputs=32, kernel_size=Gkernel_size, stride=1, dilation=4,
                                      padding=8, dropout=dropout)
        self.up_sampling2 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.Decoder2 = TemporalBlock(n_inputs=32, n_outputs=16, kernel_size=Gkernel_size, stride=1, dilation=2,
                                      padding=4, dropout=dropout)
        # self.Decoder3 = TemporalBlock(n_inputs=16, n_outputs=1, kernel_size=Gkernel_size, stride=1, dilation=1,
        #                                 padding=2, dropout=dropout)
        self.flatten = nn.Flatten()
        self.FC = nn.Linear(in_features=3200, out_features=200)
        self.apply(self.G_initialize_weights)

    def forward(self, x):
        # """
        # Input parameter:
        # x: shape: (batch_size, nun_steps，input_size=1) or (batch_size, input_size, num_steps)
        # Output parameter:
        # output: Generated FECG, shape: (num_steps, batch_size, input_size=1)
        # """
        if x.shape[2] == 1:
            x = x.permute(0, 2, 1)
        x1 = self.Encoder1(x)
        x = self.down_sampling1(x1)
        x2 = self.Encoder2(x)
        x = self.down_sampling2(x2)
        x3 = self.Encoder3(x)
        y1 = self.up_sampling1(x3)
        y = torch.cat((x2, y1), dim=1)
        y2 = self.Decoder1(y)
        y2 = self.up_sampling2(y2)
        y = torch.cat((x1, y2), dim=1)
        y3 = self.Decoder2(y)
        y = self.flatten(y3)
        y = self.FC(y)
        y = torch.reshape(y, (-1, 200, 1))
        return y

    def G_initialize_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
            nn.init.normal_(m.weight, mean=0, std=0.002)
            nn.init.constant_(m.bias, 0)

class Discriminator(nn.Module):
    """
    Discriminator Network
    """
    def __init__(self, input_length, stride, Dkernel_size):
        super(Discriminator, self).__init__()

        self.tcn1 = TemporalBlock(n_inputs=1, n_outputs=8, kernel_size=3, stride=1, dilation=1,
                                      padding=2, dropout=0.2)
        self.tcn2 = TemporalBlock(n_inputs=8, n_outputs=16, kernel_size=3, stride=1, dilation=2,
                                      padding=4, dropout=0.2)
        # self.tcn3 = TemporalBlock(n_inputs=32, n_outputs=64, kernel_size=3, stride=1, dilation=4,
        #                               padding=8, dropout=0.2)
        self.flatten = nn.Flatten()
        self.FC1 = nn.Linear(in_features=3200, out_features=32)  # 12800   128
        self.FC2 = nn.Linear(in_features=32, out_features=1)
        self.Sigmoid = nn.Sigmoid()
        self.apply(self.D_initialize_weights)

    def forward(self, x):
        if x.shape[2] == 1:
            x = x.permute(0, 2, 1)
        batch_size = x.shape[0]
        x = self.tcn1(x)
        x = self.tcn2(x)
        # x = self.tcn3(x)
        x = torch.sin(x)
        output = self.flatten(x)
        output = self.FC1(output)
        output = torch.sin(output)
        output = self.FC2(output)
        validity = self.Sigmoid(output)
        return validity

    def D_initialize_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.002)
            nn.init.constant_(m.bias, 0)