import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()  # contiguous()方法用于保证张量的内存是连续的

class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        :param n_inputs: 输入通道数
        :param n_outputs: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步长，一般为 1
        :param dilation: 膨胀系数
        :param padding: 填充
        :param dropout: dropout比率，为更好的收敛
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的维度为：(batch_size, input_channel, seq_len+padding)
        self.chomp1 = Chomp1d(padding)  # 为了保证输出的维度和输入的维度一致，需要对输出进行调整
        self.relu1 = nn.ReLU()  # 激活函数
        self.dropout1 = nn.Dropout(dropout)  # dropout层，为更好的收敛

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None   # 为了保证输出的维度和输入的维度一致，需要对输入进行调整
        self.relu = nn.ReLU()
        self.init_weights()

    # 初始化权重
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)  # 初始化权重，均值为0，标准差为0.01的标准差的随机值
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)  # 为了保证输出的维度和输入的维度一致，需要对输入进行调整
        return self.relu(out + res)

# ------------------------------------TCN------------------------------------
# 此类适用于构造TCN网络，会调用TemporalBlock类
# ---------------------------------------------------------------------------
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构.
        num_inputs : int 输入通道数
        num_channels: list 每层的hidden_channel数，例如：[5,12,1]表示有3个隐层，每层hidden_channel数为5、12、1
        kernel_size: int 卷积核尺寸
        dropout: float 丢弃的比率

        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: shape为 (Batch_size, input_channel, seq_len)
        return： shape为(Batch_size, outpout_channel, seq_len)
        """
        return self.network(x)
