from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class ADEOutdoorDataset(CocoDataset):

    CLASSES = ('sky', 'ground', 'water', 'building', 'billboard', 'mountain',
               'vegetation', 'vehicle', 'person', 'other')