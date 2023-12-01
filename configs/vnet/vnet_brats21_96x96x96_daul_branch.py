from mmengine.config import read_base
from monai.losses import DiceLoss
from seg.models.unet.monai_unet_mod import UNetMod
from seg.models.segmentors.monai_model import MonaiSeg
from seg.models.medical_seg.vnet_dtf import VNet
from seg.engine.hooks.set_epoch_hook import SetEpochInfoHook
from seg.models.segmentors.monai_dual_model import MonaiDualSeg, MSELoss, AdvMSELoss

with read_base():
    from .._base_.datasets.brats21 import *  # noqa
    from .._base_.schedules.schedule_100e_sgd import *  # noqa
    from .._base_.monai_runtime import *  # noqa


# model settings
model = dict(
    type=MonaiDualSeg,
    num_classes=4,
    roi_shapes=roi,
    backbone=dict(
        type=VNet,
        spatial_dims=3,
        in_channels=4,
        out_channels=3),
    loss_functions=[
        dict(type=DiceLoss, to_onehot_y=False, sigmoid=True),
        dict(type=MSELoss, steady_point=100, loss_weight=2.0),
                    ],
    infer_cfg=dict(
        inf_size=roi,
        sw_batch_size=4,    # number of sliding window batch size
        infer_overlap=0.5   # sliding window inference overlap
    ))

default_hooks.update(
    dict(
        logger=dict(interval=10, val_interval=50)))

val_cfg = dict(type=MonaiValLoop, print_log_per_case=False)
test_cfg = dict(type=MonaiTestLoop, print_log_per_case=False)
# custom hooks
custom_hooks = [dict(type=SetEpochInfoHook)]
vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='brats21', name='unet-tiny-sgd-100e'),
        define_metric_cfg=dict(Dice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
work_dir = '../working_brats21/SGD_100epochs/vnet/vnet-monai-dualBranch-DiceLoss'