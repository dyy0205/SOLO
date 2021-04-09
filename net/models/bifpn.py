import torch
from torch import nn
# from converter_helper import functional as F
try:
    from models import gen_efficientnet, bifpn
    from converter_helper.main import functional as F
except ModuleNotFoundError:
    # from net2.smooth_cross_entropy_loss import SmoothCrossEntropyLoss
    from net.models import gen_efficientnet, bifpn
    from net.converter_helper.main import functional as F


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=kernel_size // 2, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


class SeparableConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2d(in_channels, in_channels,
                                     kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.relu(x)

        return x


class SingleBiFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 first_block=False,
                 epsilon=1e-4,
                 attention=True):
        """
        Args:
            first_block: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
        """
        super(SingleBiFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.first_block = first_block
        self.epsilon = epsilon
        self.attention = attention

        if self.first_block:
            self.p2_down_channel = nn.Sequential(
                    Conv2d(in_channels[0], out_channels, 1),
                    nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
                )
            self.p3_down_channel = nn.Sequential(
                    Conv2d(in_channels[1], out_channels, 1),
                    nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                    Conv2d(in_channels[2], out_channels, 1),
                    nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel = nn.Sequential(
                    Conv2d(in_channels[3], out_channels, 1),
                    nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                    Conv2d(in_channels[3], out_channels, 1),
                    nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
                    nn.MaxPool2d(3, 2, padding=1)
            )

            self.p3_down_channel_2 = nn.Sequential(
                Conv2d(in_channels[1], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel_2 = nn.Sequential(
                Conv2d(in_channels[2], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2d(in_channels[3], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )

        # Conv layers
        self.conv5_up = SeparableConvBlock(out_channels)
        self.conv4_up = SeparableConvBlock(out_channels)
        self.conv3_up = SeparableConvBlock(out_channels)
        self.conv2_up = SeparableConvBlock(out_channels)
        self.conv3_down = SeparableConvBlock(out_channels)
        self.conv4_down = SeparableConvBlock(out_channels)
        self.conv5_down = SeparableConvBlock(out_channels)
        self.conv6_down = SeparableConvBlock(out_channels)

        # top-down (upsample to target phase's by nearest interpolation)
        # self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')
        self.p4_upsample = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')
        self.p3_upsample = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')
        self.p2_upsample = lambda x: F.interpolate(x, scale_factor=2, mode='nearest')

        # bottom-up (downsample to target phase's by pooling)
        self.p3_downsample = nn.MaxPool2d(3, 2, padding=1)
        self.p4_downsample = nn.MaxPool2d(3, 2, padding=1)
        self.p5_downsample = nn.MaxPool2d(3, 2, padding=1)
        self.p6_downsample = nn.MaxPool2d(3, 2, padding=1)

        self.relu = nn.ReLU()

        # Weight
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.ReLU()

        self.p3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P6_0 -------------------------> P6_2 -------->
               |-------------|                ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P3_0 ---------> P3_1 ---------> P3_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P2_0 -------------------------> P2_2 -------->
        """

        if self.attention:
            p2_out, p3_out, p4_out, p5_out, p6_out = self._forward_fast_attention(inputs)
        else:
            p2_out, p3_out, p4_out, p5_out, p6_out = self._forward(inputs)

        return p2_out, p3_out, p4_out, p5_out, p6_out

    def _forward_fast_attention(self, inputs):
        if self.first_block:
            p2, p3, p4, p5 = inputs
            p2_in = self.p2_down_channel(p2)
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
            p6_in = self.p5_to_p6(p5)
        else:
            p2_in, p3_in, p4_in, p5_in, p6_in = inputs

        # P6_0 to P6_2

        # Weights for P5_0 and P6_0 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)   # to numpy
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(self.relu(float(weight[0]) * p5_in + float(weight[1]) * self.p5_upsample(p6_in)))

        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p4_up = self.conv4_up(self.relu(float(weight[0]) * p4_in + float(weight[1]) * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_0 to P3_1
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_0 to P3_1 respectively
        p3_up = self.conv3_up(self.relu(float(weight[0]) * p3_in + float(weight[1]) * self.p3_upsample(p4_up)))

        # Weights for P2_0 and P3_1 to P2_2
        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        # Connections for P2_0 and P3_1 to P2_2 respectively
        p2_out = self.conv2_up(self.relu(float(weight[0]) * p2_in + float(weight[1]) * self.p2_upsample(p3_up)))

        if self.first_block:
            p3_in = self.p3_down_channel_2(p3)
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P3_0, P3_1 and P2_2 to P3_2
        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        # Connections for P3_0, P3_1 and P2_2 to P3_2 respectively
        p3_out = self.conv3_down(
            self.relu(float(weight[0]) * p3_in + float(weight[1]) * p3_up + float(weight[2]) * self.p3_downsample(p2_out)))

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.relu(float(weight[0]) * p4_in + float(weight[1]) * p4_up + float(weight[2]) * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.relu(float(weight[0]) * p5_in + float(weight[1]) * p5_up + float(weight[2]) * self.p5_downsample(p4_out)))

        # Weights for P6_0 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0 and P5_2 to P6_2
        p6_out = self.conv6_down(self.relu(float(weight[0]) * p6_in + float(weight[1]) * self.p6_downsample(p5_out)))

        return p2_out, p3_out, p4_out, p5_out, p6_out

    def _forward(self, inputs):
        if self.first_block:
            p2, p3, p4, p5 = inputs
            p2_in = self.p2_down_channel(p2)
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
            p6_in = self.p5_to_p6(p5)
        else:
            p2_in, p3_in, p4_in, p5_in, p6_in = inputs

        # P6_0 to P6_2

        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(self.relu(p5_in + self.p5_upsample(p6_in)))

        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.relu(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_0 to P3_1 respectively
        p3_up = self.conv3_up(self.relu(p3_in + self.p3_upsample(p4_up)))

        # Connections for P2_0 and P3_1 to P2_2 respectively
        p2_out = self.conv2_up(self.relu(p2_in + self.p2_upsample(p3_up)))

        if self.first_block:
            p3_in = self.p3_down_channel_2(p3)
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P3_0, P3_1 and P2_2 to P3_2 respectively
        p3_out = self.conv3_down(
            self.relu(p3_in + p3_up + self.p3_downsample(p2_out)))

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.relu(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.relu(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0 and P5_2 to P6_2
        p6_out = self.conv6_down(self.relu(p6_in + self.p6_downsample(p5_out)))

        return p2_out, p3_out, p4_out, p5_out, p6_out


class BiFPN_Lite(nn.Module):

    def __init__(self,
                 compound_coef=0,
                 num_repeats=None,
                 out_channels=None,
                 num_classes=None):
        super(BiFPN_Lite, self).__init__()
        self.compound_coef = compound_coef
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.conv_channel_coef = {
            # the channels of C2/C3/C4/C5.
            0: [24, 40, 112, 320],
            1: [24, 40, 112, 320],
            2: [24, 48, 120, 352],
            3: [32, 48, 136, 384],
            4: [32, 56, 160, 448],
            5: [40, 64, 176, 512],
            6: [40, 72, 200, 576],
            7: [40, 72, 200, 576],
        }
        self.num_repeats = num_repeats if num_repeats is not None else self.fpn_cell_repeats[compound_coef]
        self.out_channels = out_channels if out_channels is not None else self.fpn_num_filters[self.compound_coef]

        self.bifpn = nn.Sequential(
            *[SingleBiFPN(in_channels=self.conv_channel_coef[compound_coef],
                          out_channels=self.out_channels,
                          first_block=True if _ == 0 else False,
                          attention=True if compound_coef < 6 else False)
              for _ in range(self.num_repeats)])

        self.class_conv = nn.Conv2d(self.fpn_num_filters[compound_coef], num_classes, kernel_size=3, padding=1,
                                    bias=False)

    def forward(self, inputs):
        # print([inputs[i].shape for i in range(len(inputs))])
        feats = self.bifpn(inputs)
        outs = []
        for i in range(len(feats) - 1, -1, -1):
            outs.append(self.class_conv(feats[i]))
        return outs
