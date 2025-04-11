# BoneMNet包初始化文件
"""
BoneMNet - 骨骼形态学网络包
用于骨骼细胞形态学分析和分类的深度学习工具包
"""

from .core import BoneMorphoNetModel, BoneMorphoNetTrainer, BoneMorphoNetPredictor
from .data_augmentation import AutoAugment, CutMix
from .utils import BoneMorphoNetUtils

__version__ = '0.1.0'
__author__ = 'BoneMNet Team'

__all__ = [
    'BoneMorphoNetModel',
    'BoneMorphoNetTrainer',
    'BoneMorphoNetPredictor',
    'AutoAugment',
    'CutMix',
    'BoneMorphoNetUtils'
]