import math
# from re import X
from statistics import mode
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule
from mmcv.cnn import Conv3d
from torch.nn import InstanceNorm3d, LeakyReLU, Softplus
from monai.networks.blocks.convolutions import Convolution



# class Global_T(nn.Module):
#     def __init__(self):
#         super(Global_T, self).__init__()
#         self.mlp = InstanceTemperature()
#         # self.global_T = nn.Parameter(mlp)
#         self.grl = GradientReversal()

#     def forward(self, fake_input1, fake_input2, lambda_):
#         return self.grl(self.mlp(fake_input1,fake_input2), lambda_)

class Global_T(nn.Module):
    def __init__(self):
        super(Global_T, self).__init__()
        
        self.global_T = nn.Parameter(torch.ones(1), requires_grad=True)
        self.grl = GradientReversal()

    def forward(self, fake_input1, fake_input2, lambda_):
        return self.grl(self.global_T, lambda_)
    
class InstanceTemperature(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(InstanceTemperature, self).__init__()
        
        self.input_dim = input_dim * 2 
        self.spatial_dims = 3
        self.conv = nn.Sequential()
        unit = Convolution(
                self.spatial_dims,
                input_dim,
                output_dim,
                kernel_size=(1, 1, 1),
                padding=(0, 0, 0),
            )
        self.conv.add_module(f"unit_0", unit)
        unit = Convolution(
                self.spatial_dims,
                output_dim,
                output_dim,
                kernel_size=(1, 1, 3),
                padding=(0, 0, 1),
            )
        self.conv.add_module(f"unit_1", unit)
        unit = Convolution(
                self.spatial_dims,
                output_dim,
                output_dim,
                kernel_size=(1, 1, 3),
                padding=(0, 0, 1),
            )
        self.conv.add_module(f"unit_2", unit)
        unit = Convolution(
                self.spatial_dims,
                output_dim,
                output_dim,
                kernel_size=(1, 1, 3),
                padding=(0, 0, 1),
            )
        self.conv.add_module(f"unit_3", unit)
        # 用全局平均池化
        self.global_pooling = nn.AdaptiveAvgPool3d(1)

    def forward(self,seg_tensor, dis_tensor):
        concat_tensor = torch.cat([seg_tensor, dis_tensor], dim=1)
        temperature = self.conv(concat_tensor)
        temperature = self.global_pooling(temperature)
        return temperature


from torch.autograd import Function
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        # print(dx)
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
        # self.lambda_ = lambda_

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)