# Copyright (c) HEMEvidence. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from seg.registry import MODELS
from .utils.epoch_decay import CosineDecay, LinearDecay
from .utils.temp_gloabal import Global_T
from .mse_loss import get_current_consistency_weight


@MODELS.register_module()
class AdvMSELoss(nn.Module):
    """
    Transitioning from the original MSE Loss to the adversarial MSE Loss, 
    the loss function

    Args:
        nn (_type_): _description_
        loss_weight(float): Loss weights are the maximum weights to control against losses
        steady_point(int): Smoothing of loss weights from 0 to 1 for raw MSE Loss
        first_point(int): The first inflection point of the entire excessive loss
        second_point(int): Second point of inflection of the entire excess of loss
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_mse'.
    """
    def __init__(self,
                 loss_weight1: float = 1.0,
                 loss_weight2: float = 1.0,
                 steady_point: int = 50,
                 first_point: int = 10,
                 second_point: int = 20,
                 third_point: int = 500,
                 fourth_point: int = 500,
                 loss_name: str = 'loss_adversarial_mse'):
        super().__init__()
        self.loss_weight1 = loss_weight1
        self.loss_weight2 = loss_weight2
        self.loss_name_ = loss_name
        self.first_point = first_point
        self.steady_point = steady_point
        self.second_point = second_point
        self.third_point = third_point
        self.fourth_point = fourth_point

    def forward(self, outputs: Tensor, dis_to_mask: Tensor, current_epoch) -> Tensor:
       
        outputs_soft = outputs.sigmoid()
        # mse = (dis_to_mask - outputs_soft) ** 2
        mse_loss_func = torch.nn.MSELoss(reduction='none')
        mse = mse_loss_func(outputs_soft.float(), dis_to_mask.float())

        if current_epoch < self.first_point:
            # Gradual increase in MSE losses
            ramps = get_current_consistency_weight(current_epoch, self.steady_point)
            mse_loss = self.loss_weight1 * ramps * mse
            loss = torch.mean(mse_loss)   
            return loss
        else:
            if current_epoch < self.second_point:
                a = 1.0 - (current_epoch-self.first_point) / (self.second_point-self.first_point)
            elif current_epoch < self.third_point:
                a = 0.0
            elif current_epoch < self.fourth_point:
                a = (current_epoch-self.first_point) / (self.second_point-self.first_point)
            else:
                a = 1.0
            gradient_decay = CosineDecay(max_value=0, min_value=-1, num_loops=10)
            decay_value = gradient_decay.get_value(current_epoch)
            mlp_net = Global_T()
            mlp_net.cuda()
            mlp_net.train()
            temp = mlp_net(dis_to_mask, outputs_soft, decay_value)
            temp = 1 + 20 * torch.sigmoid(temp)
            adv_mse = 0.5 * (mse / temp) 
            # adv_mse = 0.5 *((mse / temp) + torch.log(1.0 + temp))
            mse_loss = a * self.loss_weight1 * mse +  (1.0 - a) * self.loss_weight2 * adv_mse
            loss = torch.mean(mse_loss)
            return loss
       
            

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self.loss_name_func
