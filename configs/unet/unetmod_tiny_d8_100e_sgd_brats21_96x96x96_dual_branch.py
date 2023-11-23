from mmengine.config import read_base
from monai.losses.dice import DiceCELoss, DiceLoss, DiceFocalLoss
from monai.networks.nets import UNet
from seg.models.segmentors.monai_model import MonaiSeg
from monai.losses.focal_loss import FocalLoss
from seg.models.segmentors.monai_dual_model import MonaiDualSeg, MSELoss
from seg.models.decode_heads.dual_branch import DualBranchRes, DualBranchTanh
from seg.engine.hooks.set_epoch_hook import SetEpochInfoHook
from seg.models.monai_datapreprocessor import MonaiBratsPreProcessor

with read_base():
    from .._base_.datasets.brats19 import *  # noqa
    from .._base_.schedules.schedule_100e_sgd import *  # noqa
    from .._base_.monai_runtime import *  # noqa

# model settings
model = dict(
    type=MonaiDualSeg,
    num_classes=4,
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
    decoder=dict(
        type=DualBranchRes,
        spatial_dims=3,
        out_channels=1
    ),
    loss_functions=[
        dict(type=DiceFocalLoss, to_onehot_y=False, sigmoid=True, squared_pred=True, include_background=True),
        dict(type=DiceLoss, to_onehot_y=False, sigmoid=False, squared_pred=True, include_background=True),
        dict(type=MSELoss, steady_point=100)
        ],
    data_preprocessor=dict(type=MonaiBratsPreProcessor),
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

val_cfg = dict(type=MonaiValLoop, print_log_per_case=False)
test_cfg = dict(type=MonaiTestLoop, print_log_per_case=False)

vis_backends = [
    dict(type=LocalVisBackend),
    dict(
        type=WandbVisBackend,
        init_kwargs=dict(
            project='brats19', name='unet-tiny-sgd-100e'),
        define_metric_cfg=dict(Dice='max'))
]
visualizer = dict(type=SegLocalVisualizer,
                  vis_backends=vis_backends,
                  name='visualizer')

work_dir = '../working_brats19/SGD_200epochs/unet_num_res_2/resuent-resdaul_branch-monai-FocalDiceLoss-MSE'