from seg.evaluation.metrics.monai_metric import MonaiMetric
from seg.evaluation.monai_evaluator import MonaiEvaluator
from seg.datasets.monai_dataset import BRATS19_METAINFO
roi = [96,96,96]

dataloader_cfg = dict(
    data_name='BraTS19',
    # data_path='/home/jz207/workspace/wutq/UG-MCL/data/BraTS2019',
    data_path='/home/jz207/workspace/data/brats21/',
    patch_size=[96,96,96],
    batch_size=8,
    workers=8,
    meta_info=BRATS19_METAINFO,
)

val_evaluator = dict(
    type=MonaiEvaluator,
    metrics=dict(
        type=MonaiMetric,
        metrics=['Dice', 'HD95'],
        num_classes=1,
        one_hot=False,
        include_background=True,
        reduction='mean_batch',
        print_per_class=False))
test_evaluator = val_evaluator