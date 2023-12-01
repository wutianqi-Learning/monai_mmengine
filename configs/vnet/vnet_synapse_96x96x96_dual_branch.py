from mmengine.config import read_base
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.losses.dice import DiceCELoss, DiceLoss, DiceFocalLoss
from seg.models.segmentors.monai_dual_model import MonaiDualSeg, MSELoss
from seg.models.segmentors.monai_model import MonaiSeg
from seg.models.medical_seg.vnet_dtf import VNet
from seg.engine.hooks.set_epoch_hook import SetEpochInfoHook

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
        dict(type=MSELoss, steady_point=1000)
    ],
    infer_cfg=dict(
        inf_size=roi,
        sw_batch_size=4,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))
# custom hooks
custom_hooks = [dict(type=SetEpochInfoHook)]
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='synapse', name='unet-base-sgd-1000e'),
        define_metric_cfg=dict(loss='min'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
work_dir = '../working_synapse/SGD_1000epochs/vnet/vnet-monai-dualBranch-DiceCELoss-MSE'