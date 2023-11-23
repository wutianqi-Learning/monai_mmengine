
from __future__ import annotations
import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution
from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers.factories import Conv


class DualBranchTanh(nn.Module):
    def __init__(
        self
    ) -> None:
        """
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        """
        super().__init__()
        self.tanh = nn.Tanh()
        
    def forward(self, output: torch.Tensor) -> torch.Tensor:
        dis_out = self.tanh(output)
        return output, dis_out
    

class ResidualUnit(nn.Module):
    """
    Residual module with multiple convolutions and a residual connection.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        subunits: int = 2,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Sequence[int] | int | None = None,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        self.residual = nn.Identity()
        if not padding:
            padding = same_padding(kernel_size, dilation)
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)

        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = Convolution(
                self.spatial_dims,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=(1, 1, 3),
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=(0, 0, 1),
            )

            self.conv.add_module(f"unit{su:d}_0", unit)
            
            unit = Convolution(
                self.spatial_dims,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=(1, 3, 1),
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=(0, 1, 0),
            )

            self.conv.add_module(f"unit{su:d}_1", unit)
            
            unit = Convolution(
                self.spatial_dims,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=(3, 1, 1),
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=(1, 0, 0),
            )

            self.conv.add_module(f"unit{su:d}_2", unit)

            # after first loop set channels and strides to what they should be for subsequent units
            schannels = out_channels
            sstrides = 1

        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size = kernel_size
            rpadding = padding

            if np.prod(strides) == 1:  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernel_size = 1
                rpadding = 0

            conv_type = Conv[Conv.CONV, self.spatial_dims]
            self.residual = conv_type(in_channels, out_channels, rkernel_size, strides, rpadding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = self.residual(x)  # create the additive residual from x
        cx: torch.Tensor = self.conv(x)  # apply x to sequence of operations
        return cx + res  # add the residual to the output
    
class DualBranchRes(nn.Module):
    def __init__(
        self,
        spatial_dims,
        out_channels) -> None:
        """
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels
        self.convs = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            adn_ordering="AN",
            dropout=0.0,
            subunits=1
        )

        self.tanh = nn.Tanh()

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        #  make residual
        dis_out = self.convs(output)
        dis_out = self.tanh(dis_out)
        return output, dis_out
    

    
