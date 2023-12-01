from mmengine.config import read_base
from monai.losses.dice import DiceCELoss, DiceLoss, DiceFocalLoss
from seg.models.segmentors.monai_dual_model import MonaiDualSeg, MSELoss
from monai.networks.nets import UNet
from seg.models.segmentors.monai_model import MonaiSeg
from monai.losses.focal_loss import FocalLoss

from seg.models.decode_heads.dual_branch import DualBranchRes, DualBranchTanh
from seg.engine.hooks.set_epoch_hook import SetEpochInfoHook
from seg.models.monai_datapreprocessor import MonaiBratsPreProcessor
from seg.models.medical_seg.vnet_dtf import VNet

with read_base():
    from .._base_.datasets.synapse import *  # noqa
    from .._base_.schedules.schedule_1000e_sgd import *  # noqa
    from .._base_.monai_runtime import *  # noqa

# model settings
model = dict(
    type=MonaiDualSeg,
    num_classes=14,
    roi_shapes=roi,
    backbone=dict(
        type=VNet,
        spatial_dims=3,
        in_channels=1,
        out_channels=14),
    loss_functions=[
        dict(type=DiceCELoss, to_onehot_y=True, softmax=True),
        dict(type=DiceLoss, to_onehot_y=True),
        dict(type=MSELoss, steady_point=500)
    ],
    # data_preprocessor=dict(type=MonaiBratsPreProcessor),
    infer_cfg=dict(
        inf_size=roi,
        sw_batch_size=4,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))
# custom hooks
custom_hooks = [dict(type=SetEpochInfoHook)]
default_hooks.update(
    dict(
        logger=dict(interval=10, val_interval=50)))

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='btcv', name='vnet-tiny-sgd-1000e'),
        define_metric_cfg=dict(Dice='min'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')

work_dir = '../working_btcv/SGD_1000epochs/vnet/vnet-daul_branch-monai-DiceLoss-MSE'