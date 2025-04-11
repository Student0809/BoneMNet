# BoneMNet数据增强模块初始化文件
"""
BoneMNet数据增强模块
包含各种数据增强方法和工具，如AutoAugment、RandAugment、AugMix和CutMix等
"""

from .auto_augment import rand_augment_transform as AutoAugment
from .auto_augment import AutoAugment as AutoAugmentClass
from .auto_augment import RandAugment
from .auto_augment import AugMixAugment
from .CutMix import Mixup, FastCollateMixup
from .augmentations import (
    get_auto_augment_transform,
    get_cutmix_transform,
    get_fast_cutmix_transform,
    get_standard_transforms,
    get_advanced_transforms
)

__version__ = '0.1.0'
__author__ = 'BoneMNet Team'

__all__ = [
    'AutoAugment',
    'AutoAugmentClass',
    'RandAugment',
    'AugMixAugment',
    'Mixup',
    'FastCollateMixup',
    'get_auto_augment_transform',
    'get_cutmix_transform',
    'get_fast_cutmix_transform',
    'get_standard_transforms',
    'get_advanced_transforms'
]