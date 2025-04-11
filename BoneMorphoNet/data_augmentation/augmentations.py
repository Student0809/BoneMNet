"""
BoneMNet数据增强便捷接口
提供各种数据增强方法的便捷使用方式
"""

from typing import Dict, List, Optional, Union, Tuple
import torch
from torchvision import transforms
from PIL import Image

from .auto_augment import rand_augment_transform, auto_augment_transform, augment_and_mix_transform
from .CutMix import Mixup, FastCollateMixup

def get_auto_augment_transform(config_str: str = 'rand-m9-n2-mstd0.5', hparams: Optional[Dict] = None):
    """
    获取AutoAugment变换
    
    Args:
        config_str: 配置字符串，默认为'rand-m9-n2-mstd0.5'
        hparams: 超参数字典
        
    Returns:
        AutoAugment变换函数
    """
    if hparams is None:
        hparams = {}
    return rand_augment_transform(config_str, hparams)

def get_cutmix_transform(mixup_alpha: float = 1.0, cutmix_alpha: float = 0.0, 
                         cutmix_minmax: Optional[List[float]] = None, 
                         prob: float = 1.0, 
                         switch_prob: float = 0.5,
                         mode: str = 'batch', 
                         correct_lam: bool = True, 
                         label_smoothing: float = 0.1, 
                         num_classes: int = 1000):
    """
    获取CutMix变换
    
    Args:
        mixup_alpha: mixup的alpha参数
        cutmix_alpha: cutmix的alpha参数
        cutmix_minmax: cutmix的最小最大值
        prob: 应用变换的概率
        switch_prob: 切换变换的概率
        mode: 变换模式，可选'batch', 'pair', 'elem'
        correct_lam: 是否修正lambda值
        label_smoothing: 标签平滑参数
        num_classes: 类别数量
        
    Returns:
        CutMix变换对象
    """
    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        cutmix_minmax=cutmix_minmax,
        prob=prob,
        switch_prob=switch_prob,
        mode=mode,
        correct_lam=correct_lam,
        label_smoothing=label_smoothing,
        num_classes=num_classes
    )

def get_fast_cutmix_transform(mixup_alpha: float = 1.0, cutmix_alpha: float = 0.0, 
                             cutmix_minmax: Optional[List[float]] = None, 
                             prob: float = 1.0, 
                             switch_prob: float = 0.5,
                             mode: str = 'batch', 
                             correct_lam: bool = True, 
                             label_smoothing: float = 0.1, 
                             num_classes: int = 1000):
    """
    获取快速CutMix变换
    
    Args:
        mixup_alpha: mixup的alpha参数
        cutmix_alpha: cutmix的alpha参数
        cutmix_minmax: cutmix的最小最大值
        prob: 应用变换的概率
        switch_prob: 切换变换的概率
        mode: 变换模式，可选'batch', 'pair', 'elem'
        correct_lam: 是否修正lambda值
        label_smoothing: 标签平滑参数
        num_classes: 类别数量
        
    Returns:
        快速CutMix变换对象
    """
    return FastCollateMixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        cutmix_minmax=cutmix_minmax,
        prob=prob,
        switch_prob=switch_prob,
        mode=mode,
        correct_lam=correct_lam,
        label_smoothing=label_smoothing,
        num_classes=num_classes
    )

def get_standard_transforms(img_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    获取标准数据变换
    
    Args:
        img_size: 图像大小
        
    Returns:
        包含训练和验证变换的字典
    """
    # 训练变换
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.25, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 验证变换
    val_transform = transforms.Compose([
        transforms.Resize(int(236)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return {
        "train": train_transform,
        "val": val_transform
    }

def get_advanced_transforms(img_size: int = 224, auto_augment_config: str = 'rand-m9-n2-mstd0.5') -> Dict[str, transforms.Compose]:
    """
    获取高级数据变换，包含AutoAugment
    
    Args:
        img_size: 图像大小
        auto_augment_config: AutoAugment配置
        
    Returns:
        包含训练和验证变换的字典
    """
    # 获取AutoAugment变换
    auto_augment = get_auto_augment_transform(auto_augment_config, hparams={})
    
    # 训练变换
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.25, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        auto_augment,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 验证变换
    val_transform = transforms.Compose([
        transforms.Resize(int(236)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return {
        "train": train_transform,
        "val": val_transform
    }

__all__ = [
    'get_auto_augment_transform',
    'get_cutmix_transform',
    'get_fast_cutmix_transform',
    'get_standard_transforms',
    'get_advanced_transforms'
] 