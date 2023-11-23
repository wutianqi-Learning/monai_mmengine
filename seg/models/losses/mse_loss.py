import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, steady_point):
    consistency = 1.0
    consistency_rampup = steady_point
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)


class MSELoss(nn.Module):

    def __init__(self,
                 steady_point: int = 100,
                 loss_weight: float = 1.0,
                 loss_name: str = 'loss_mse',
                 dis_softmax: bool = False):
        super().__init__()
        self.steady_point = steady_point
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name
        self.dis_softmax = dis_softmax
        self.MSE_Loss = torch.nn.MSELoss(reduction='none')
        

    def forward(self, outputs: Tensor, dis_to_mask: Tensor, current_epoch) -> Tensor:
        outputs = torch.sigmoid(outputs)
        consistency_weight = get_current_consistency_weight(current_epoch, self.steady_point)
        cross_task_dist = torch.mean((dis_to_mask - outputs) ** 2) 
        cross_task_loss = consistency_weight * cross_task_dist
        return self.loss_weight * cross_task_loss

    @property
    def loss_name(self):
        return self.loss_name_
