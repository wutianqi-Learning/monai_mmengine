from mmengine.config import read_base
from monai.losses import DiceLoss, DiceCELoss
from monai.networks.nets import UNet
from seg.models.segmentors.monai_model import MonaiSeg, MyDiceCELoss
from seg.models.monai_datapreprocessor import MonaiBratsPreProcessor

with read_base():
    from .._base_.datasets.brats19 import *  # noqa
    from .._base_.schedules.schedule_100e_adamw import *  # noqa
    from .._base_.monai_runtime import *  # noqa

# model settings
model = dict(
    type=MonaiSeg,
    num_classes=1,
    roi_shapes=roi,
    backbone=dict(
        type=UNet,
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=5,
        channels=(8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=2),
    loss_functions= dict(type=MyDiceCELoss, to_onehot_y=False, sigmoid=True, squared_pred=True, include_background=True),
    data_preprocessor=dict(type=MonaiBratsPreProcessor),
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

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='brats21_binary', name='unet-tiny-sgd-200e'),
        define_metric_cfg=dict(Dice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')
work_dir = '../working_brats21_binary/adamw_100epochs/resuent-bs8-monai-DiceLoss'