import torch
import torch.nn as nn
from monai.networks.layers.factories import Conv, Pool, Norm, Dropout
__all__ = ['ENet']


class ENet(nn.Module):
    """Efficient Neural Network"""

    def __init__(self,
                 nclass,
                 spatial_dims,
                 in_channels=1,
                 asymmetric=False,
                 backbone='',
                 aux=False,
                 jpu=False,
                 pretrained_base=None,
                 **kwargs):
        super(ENet, self).__init__()
        self.aux = aux
        self.initial = InitialBlock(
            in_channels=in_channels,
            out_channels=15,
            spatial_dims=spatial_dims,
            **kwargs)

        self.bottleneck1_0 = Bottleneck(spatial_dims, 16, 16, 64, downsampling=True, **kwargs)
        self.bottleneck1_1 = Bottleneck(spatial_dims, 64, 16, 64, **kwargs)
        self.bottleneck1_2 = Bottleneck(spatial_dims, 64, 16, 64, **kwargs)
        self.bottleneck1_3 = Bottleneck(spatial_dims, 64, 16, 64, **kwargs)
        self.bottleneck1_4 = Bottleneck(spatial_dims, 64, 16, 64, **kwargs)

        self.bottleneck2_0 = Bottleneck(spatial_dims, 64, 32, 128, downsampling=True, **kwargs)
        self.bottleneck2_1 = Bottleneck(spatial_dims, 128, 32, 128, **kwargs)
        self.bottleneck2_2 = Bottleneck(spatial_dims, 128, 32, 128, dilation=2, **kwargs)
        self.bottleneck2_3 = Bottleneck(spatial_dims, 128, 32, 128, asymmetric=asymmetric, **kwargs)
        self.bottleneck2_4 = Bottleneck(spatial_dims, 128, 32, 128, dilation=4, **kwargs)
        self.bottleneck2_5 = Bottleneck(spatial_dims, 128, 32, 128, **kwargs)
        self.bottleneck2_6 = Bottleneck(spatial_dims, 128, 32, 128, dilation=8, **kwargs)
        self.bottleneck2_7 = Bottleneck(spatial_dims, 128, 32, 128, asymmetric=asymmetric, **kwargs)
        self.bottleneck2_8 = Bottleneck(spatial_dims, 128, 32, 128, dilation=16, **kwargs)

        self.bottleneck3_1 = Bottleneck(spatial_dims, 128, 32, 128, **kwargs)
        self.bottleneck3_2 = Bottleneck(spatial_dims, 128, 32, 128, dilation=2, **kwargs)
        self.bottleneck3_3 = Bottleneck(spatial_dims, 128, 32, 128, asymmetric=asymmetric, **kwargs)
        self.bottleneck3_4 = Bottleneck(spatial_dims, 128, 32, 128, dilation=4, **kwargs)
        self.bottleneck3_5 = Bottleneck(spatial_dims, 128, 32, 128, **kwargs)
        self.bottleneck3_6 = Bottleneck(spatial_dims, 128, 32, 128, dilation=8, **kwargs)
        self.bottleneck3_7 = Bottleneck(spatial_dims, 128, 32, 128, asymmetric=asymmetric, **kwargs)
        self.bottleneck3_8 = Bottleneck(spatial_dims, 128, 32, 128, dilation=16, **kwargs)

        self.bottleneck4_0 = UpsamplingBottleneck(spatial_dims, 128, 16, 64, **kwargs)
        self.bottleneck4_1 = Bottleneck(spatial_dims, 64, 16, 64, **kwargs)
        self.bottleneck4_2 = Bottleneck(spatial_dims, 64, 16, 64, **kwargs)

        self.bottleneck5_0 = UpsamplingBottleneck(spatial_dims, 64, 4, 16, **kwargs)
        self.bottleneck5_1 = Bottleneck(spatial_dims, 16, 4, 16, **kwargs)

        self.fullconv = Conv[Conv.CONVTRANS, spatial_dims](16, nclass, 2, 2, bias=False)

        self.__setattr__('exclusive', ['bottleneck1_0', 'bottleneck1_1', 'bottleneck1_2', 'bottleneck1_3',
                                       'bottleneck1_4', 'bottleneck2_0', 'bottleneck2_1', 'bottleneck2_2',
                                       'bottleneck2_3', 'bottleneck2_4', 'bottleneck2_5', 'bottleneck2_6',
                                       'bottleneck2_7', 'bottleneck2_8', 'bottleneck3_1', 'bottleneck3_2',
                                       'bottleneck3_3', 'bottleneck3_4', 'bottleneck3_5', 'bottleneck3_6',
                                       'bottleneck3_7', 'bottleneck3_8', 'bottleneck4_0', 'bottleneck4_1',
                                       'bottleneck4_2', 'bottleneck5_0', 'bottleneck5_1', 'fullconv'])

    def forward(self, x):
        # init
        low_feature = self.initial(x)

        # stage 1
        x, max_indices1 = self.bottleneck1_0(low_feature)
        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)

        # stage 2
        x, max_indices2 = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)

        # stage 3
        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)
        x = self.bottleneck3_8(x)
        high_feature = x

        # stage 4
        x = self.bottleneck4_0(x, max_indices2)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        # stage 5
        x = self.bottleneck5_0(x, max_indices1)
        x = self.bottleneck5_1(x)

        # out
        x = self.fullconv(x)
        if self.aux:
            return x, low_feature, high_feature
        else:
            return x


class InitialBlock(nn.Module):
    """ENet initial block"""

    def __init__(self, in_channels, out_channels, spatial_dims, norm_layer=nn.BatchNorm2d, **kwargs):
        super(InitialBlock, self).__init__()
        self.conv = Conv[Conv.CONV, spatial_dims](in_channels, out_channels, 3, 2, 1, bias=False)
        self.maxpool = Pool[Pool.MAX, spatial_dims](2, 2)
        self.bn = Norm[Norm.BATCH, spatial_dims](out_channels + 1)
        self.act = nn.PReLU()

    def forward(self, x):
        # x = torch.cat([x, x, x], dim=1)  # 扩充为3通道
        x_conv = self.conv(x)
        x_pool = self.maxpool(x)
        x = torch.cat([x_conv, x_pool], dim=1)
        x = self.bn(x)
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    """Bottlenecks include regular, asymmetric, downsampling, dilated"""

    def __init__(self, spatial_dims, in_channels, inter_channels, out_channels, dilation=1, asymmetric=False,
                 downsampling=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(Bottleneck, self).__init__()
        self.downsamping = downsampling
        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_layer = Norm[Norm.BATCH, spatial_dims]
        if downsampling:
            self.maxpool = Pool[Pool.MAX, spatial_dims](2, 2, return_indices=True)
            self.conv_down = nn.Sequential(
                conv_type(in_channels, out_channels, 1, bias=False),
                norm_layer(out_channels)
            )

        self.conv1 = nn.Sequential(
            conv_type(in_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.PReLU()
        )

        if downsampling:
            self.conv2 = nn.Sequential(
                conv_type(inter_channels, inter_channels, 2, stride=2, bias=False),
                norm_layer(inter_channels),
                nn.PReLU()
            )
        else:
            if asymmetric:
                if spatial_dims == 2:
                    self.conv2 = nn.Sequential(
                        conv_type(inter_channels, inter_channels, (5, 1), padding=(2, 0), bias=False),
                        conv_type(inter_channels, inter_channels, (1, 5), padding=(0, 2), bias=False),
                        norm_layer(inter_channels),
                        nn.PReLU()
                    )
                else:
                    self.conv2 = nn.Sequential(
                        conv_type(inter_channels, inter_channels, (5, 5, 1), padding=(2, 2, 0), bias=False),
                        conv_type(inter_channels, inter_channels, (1, 1, 5), padding=(0, 0, 2), bias=False),
                        norm_layer(inter_channels),
                        nn.PReLU()
                    )
            else:
                self.conv2 = nn.Sequential(
                    conv_type(inter_channels, inter_channels, 3, dilation=dilation, padding=dilation, bias=False),
                    norm_layer(inter_channels),
                    nn.PReLU()
                )
        self.conv3 = nn.Sequential(
            conv_type(inter_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            Dropout[Dropout.DROPOUT, spatial_dims](0.1)
        )
        self.act = nn.PReLU()

    def forward(self, x):
        identity = x
        if self.downsamping:
            identity, max_indices = self.maxpool(identity)
            identity = self.conv_down(identity)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.act(out + identity)

        if self.downsamping:
            return out, max_indices
        else:
            return out


class UpsamplingBottleneck(nn.Module):
    """upsampling Block"""

    def __init__(self, spatial_dims, in_channels, inter_channels, out_channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(UpsamplingBottleneck, self).__init__()
        conv_type = Conv[Conv.CONV, spatial_dims]
        norm_layer = Norm[Norm.BATCH, spatial_dims]
        self.conv = nn.Sequential(
            conv_type(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )
        if spatial_dims == 2:
            self.upsampling = nn.MaxUnpool2d(2)
        else:
            self.upsampling = nn.MaxUnpool3d(2)

        self.block = nn.Sequential(
            conv_type(in_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels),
            nn.PReLU(),
            Conv[Conv.CONVTRANS, spatial_dims](inter_channels, inter_channels, 2, 2, bias=False),
            norm_layer(inter_channels),
            nn.PReLU(),
            conv_type(inter_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            Dropout[Dropout.DROPOUT, spatial_dims](0.1)
        )
        self.act = nn.PReLU()

    def forward(self, x, max_indices):
        out_up = self.conv(x)
        out_up = self.upsampling(out_up, max_indices)

        out_ext = self.block(x)
        out = self.act(out_up + out_ext)
        return out
