"""
original code from facebook research:
https://github.com/facebookresearch/ConvNeXt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import open_clip
from timm.models.layers import trunc_normal_
label_queue = [
    ["Neutrophilic stab granulocyte's cell shape is round-like.",
     "Neutrophilic stab granulocyte's nuclear shape is rod-shaped, S-shaped, or U-shaped.",
     "Neutrophilic stab granulocyte's cytoplasm is light blue.",
     "Neutrophilic stab granulocyte has neutrophilic granules in its cytoplasm."],

    ["Polychromatic normoblast's cell shape is round.",
     "Polychromatic normoblast's nuclear shape is round.",
     "Polychromatic normoblast's cytoplasm is blue-gray, gray, or gray-red.",
     "Polychromatic normoblast has no granules in its cytoplasm."],

    ["Neutrophilic myelocyte's cell shape is round-like.",
     "Neutrophilic myelocyte's nuclear shape is oval, semicircular, or slightly indented.",
     "Neutrophilic myelocyte's cytoplasm is blue or light blue.",
     "Neutrophilic myelocyte has neutrophilic granules in its cytoplasm."],

    ["Neutrophilic segmented granulocyte's cell shape is round.",
     "Neutrophilic segmented granulocyte's nuclear shape is segmented, with 2 to 5 lobes.",
     "Neutrophilic segmented granulocyte's cytoplasm is light blue.",
     "Neutrophilic segmented granulocyte has neutrophilic granules in its cytoplasm."],

    ["Lymphoblast's cell shape is regular or irregular.",
     "Lymphoblast's nuclear shape is round or irregular.",
     "Lymphoblast's cytoplasm is blue or dark blue.",
     "Lymphoblast has no granules in its cytoplasm."],

    ["Neutrophilic metamyelocyte's cell shape is round-like.",
     "Neutrophilic metamyelocyte's nuclear shape is kidney-shaped or crescent-shaped.",
     "Neutrophilic metamyelocyte's cytoplasm is light blue.",
     "Neutrophilic metamyelocyte has neutrophilic granules in its cytoplasm."],

    ["Myeloblast's cell shape is round-like.",
     "Myeloblast's nuclear shape is round.",
     "Myeloblast's cytoplasm is blue or dark blue.",
     "Myeloblast has few granules or no granules in its cytoplasm."],

    ["Orthochromatic normoblast's cell shape is round or oval.",
     "Orthochromatic normoblast's nuclear shape is round.",
     "Orthochromatic normoblast's cytoplasm is light red or gray-red.",
     "Orthochromatic normoblast has no granules in its cytoplasm."],

    ["Prelymphocyte's cell shape is regular or irregular.",
     "Prelymphocyte's nuclear shape is roughly round or irregular.",
     "Prelymphocyte's cytoplasm is blue or dark blue.",
     "Prelymphocyte has no granules in its cytoplasm."],

    ["Abnormal promyelocyte's cell shape is variable.",
     "Abnormal promyelocyte's nuclear shape is irregular, folded, twisted, or segmented.",
     "Abnormal promyelocyte's cytoplasm contains abundant purple-red granules.",
     "Abnormal promyelocyte has purple-red granules in its cytoplasm."],

    ["Monocyte's cell shape is round-like or irregular.",
     "Monocyte's nuclear shape is irregular, folded, twisted, horseshoe-shaped, or S-shaped.",
     "Monocyte's cytoplasm is light gray-blue or light blue.",
     "Monocyte has few granules in its cytoplasm."],

    ["Early normoblast's cell shape is round or oval.",
     "Early normoblast's nuclear shape is round.",
     "Early normoblast's cytoplasm is dark blue.",
     "Early normoblast has no granules in its cytoplasm."],

    ["Monoblast's cell shape is round-like or irregular.",
     "Monoblast's nuclear shape is round or irregular.",
     "Monoblast's cytoplasm is gray-blue or blue.",
     "Monoblast has few fine granules or no granules in its cytoplasm."],

    ["Promyelocyte's cell shape is round or oval.",
     "Promyelocyte's nuclear shape is round or oval.",
     "Promyelocyte's cytoplasm is blue or dark blue, containing purple-red granules.",
     "Promyelocyte has purple-red non-specific granules in its cytoplasm."],

    ["Eosinophilic segmented granulocyte's cell shape is round.",
     "Eosinophilic segmented granulocyte's nuclear shape is segmented.",
     "Eosinophilic segmented granulocyte's cytoplasm is orange-red, dark yellow, or brown.",
     "Eosinophilic segmented granulocyte has eosinophilic granules in its cytoplasm."],

    ["Eosinophilic myelocyte's cell shape is round-like.",
     "Eosinophilic myelocyte's nuclear shape is round or oval.",
     "Eosinophilic myelocyte's cytoplasm is orange-red, dark yellow, or brown.",
     "Eosinophilic myelocyte has eosinophilic granules in its cytoplasm."],

    ["Multiple myeloma cells' cell shape is irregular.",
     "Multiple myeloma cells' nuclear shape is irregular, sometimes with multiple nuclei.",
     "Multiple myeloma cells' cytoplasm contains multicolored granules.",
     "Multiple myeloma cells have multicolored granules in their cytoplasm."],

    ["Smudge cells' cell shape is irregular.",
     "Smudge cells' nuclear shape is irregular, often unclear due to fragmentation.",
     "Smudge cells' cytoplasm is enlarged and incomplete.",
     "Smudge cell has no granules in its cytoplasm (but often appears as a naked nucleus, with incomplete cytoplasm)."],

    ["Plasmacyte's cell shape is oval.",
     "Plasmacyte's nuclear shape is round or eccentrically placed, sometimes with two or more nuclei.",
     "Plasmacyte's cytoplasm is dark blue, occasionally red.",
     "Plasmacyte has few purple-red granules in its cytoplasm."],

    ["Other's cell shape is unclear.",
     "Other's nuclear shape is unclear.",
     "Other's cytoplasm is unclear.",
     "Other's granules are unclear."]
]


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

def clip_loss(logit_per_image,logit_per_text,device):
    loss_per_image = F.cross_entropy(logit_per_image,torch.arange(logit_per_image.shape[0],device=device,dtype=torch.long))
    loss_per_text = F.cross_entropy(logit_per_text,torch.arange(logit_per_text.shape[0],device=device,dtype=torch.long))
    return (loss_per_image+loss_per_text)/2
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        #print(f'qshape {q.shape}, kshape {k.shape},vshape{v.shape}')
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=768,
                 text_dim=512,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, text_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, text, visual,mixWho='visual'):
        #visual = [B,H,W,C]
        #text = [B,N,C]
        #flatten
        B, H, W, C = visual.shape
        x_flattened = visual.view(B, H * W, C)
        B, N, C = x_flattened.shape
        visual = self.memory_proj(x_flattened)
        x = self.text_proj(text)
        # print(x.shape)
        # print(visual.shape)
        if mixWho == 'visual':
            for layer in self.decoder:
                x = layer(x, visual)

        else:
            for layer in self.decoder:
                x = layer(visual, x)

        return self.out_proj(x)

class ChannelAttention(nn.Module):#通道注意力
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):#空间注意力
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6,scales=4,CBAM=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        assert dim % scales == 0
        part_channel = dim // scales

        self.depthwise_convs = nn.ModuleList() #构建DW卷积，用ModuleList这个容器来储存
        for i in range(scales - 1):
            self.depthwise_convs.append(
                nn.Conv2d(
                    in_channels=part_channel,
                    out_channels=part_channel,
                    kernel_size=7,
                    padding=3,
                    groups=part_channel #分组卷积用
                )
                #添加scales个组
            )
        self.part_channel = part_channel
        self.scales = scales

        if CBAM is True:
            self.ca = ChannelAttention(dim)
            self.sa = SpatialAttention()
        self.CBAM = CBAM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        spx = torch.split(x, self.part_channel, dim=1)
        sp = self.depthwise_convs[0](spx[0].contiguous())
        out = sp
        for i in range(1, self.scales - 1):
            sp = sp + spx[i]
            sp = self.depthwise_convs[i](sp.contiguous())
            out = torch.cat((out, sp), 1)
        x = torch.cat((out, spx[self.scales - 1]), 1)


        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.CBAM is True:
            x = x.permute(0, 3, 1, 2)
            x = self.ca(x) * x
            x = self.sa(x) * x
            x = x.permute(0, 2, 3, 1)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans: int = 3, out_chans: int = 1000,num_classes: int = 20, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        #在第3个stage和第4个stage加上CBAM
        for i in range(4):
            if i <2:
                stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value,CBAM=False)
                      for j in range(depths[i])]
                )
            else:
                stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value,CBAM=True)
                      for j in range(depths[i])]
                )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        self.fc = nn.Linear(in_features=dims[-1], out_features=out_chans)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            #nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)


        return x  # global average pooling, (N, C, H, W) -> (N, C)
    #改成
    def forward(self, x: torch.Tensor):
        z = self.forward_features(x)

        x = self.norm(z.mean([-2, -1]))
        y = self.head(x)
        proj = self.fc(x)
        
        return y, proj , z


class LayerNorm_Clip(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


#文本编码器
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm_Clip(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm_Clip(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module): #文本编码器
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

def convnext_tiny(num_classes: int,out_chans:int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes,
                     out_chans=out_chans
                     )
    return model


def convnext_small(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def convnext_base(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes)
    return model


def convnext_large(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes)
    return model


def convnext_xlarge(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes)
    return model


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v,
                 dropout=0.1, ratio=0.5):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        nn.init.xavier_normal_(self.w_qs.weight, gain=1.0)
        nn.init.xavier_normal_(self.w_ks.weight, gain=1.0)
        nn.init.xavier_normal_(self.w_vs.weight, gain=0.67)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)  # feed forward layer
        nn.init.xavier_normal_(self.fc.weight, gain=0.67)

        self.dropout = nn.Dropout(dropout)

        self.ratio = ratio

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(2 * (self.ratio * output + (1 - self.ratio) * residual))

        return output

class SelfAttnAdapter(nn.Module):

    def __init__(self, c_in, reduction=4, ratio=0.5):
        super(SelfAttnAdapter, self).__init__()
        self.attn = MultiHeadAttention(1, c_in,
                                       c_in // reduction, c_in // reduction, dropout=0.5, ratio=ratio).cuda()

    def forward(self, x):
        x = self.attn(x, x, x)
        return x


class CellLDGnet(nn.Module):
    def __init__(self,
                 embed_dim: int, #维度
                 # vision
                 num_classes, #分成几类
                 is_lock_Text = True,
                 tokenizer = None,
                 device = None,
                 ):
        super().__init__()
        transformer_width = 512
        transformer_heads = 4
        dropout = 0.1
        transformer_layers = 6

        # self.transformer = Transformer(
        #     width=transformer_width,
        #     layers=transformer_layers,
        #     heads=transformer_heads,
        #     attn_mask=self.build_attention_mask()
        # )
        self.adapter_self = SelfAttnAdapter(512, 4)

        self.MODEL_TAG = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        self.biomedclip = open_clip.create_model(self.MODEL_TAG).to(device)
        if is_lock_Text:
            self.biomedclip.lock_text_tower()
        self.transformer = self.biomedclip.text.transformer.to(device)
        self.text_proj = self.biomedclip.text.proj
        self.context_length = 256
        self.transformer_width = self.biomedclip.text.output_dim

        transformer_head = self.transformer_width // 64
        self.transformer_layers = 3
        vocab_size = self.biomedclip.text.vocab_size
        self.contextDecoder = ContextDecoder().to(device)
        # context_length: int,
        # vocab_size: int,  # （词汇表大小）是指模型可以识别和处理的独特词汇或标记（token）的数量
        # transformer_width: int,  # 文本的维度
        # transformer_heads: int,  # 多头注意力的头的数量
        # transformer_layers: int  # 几层残差注意力模块

        self.token_embedding = nn.Embedding(vocab_size, self.transformer_width)

        self.vocab_size = vocab_size
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, self.transformer_width))
        self.ln_final = LayerNorm_Clip(self.transformer_width)

        self.text_projection = nn.Parameter(torch.empty(self.transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.visual = convnext_tiny(num_classes,out_chans=embed_dim).to(device) #更改为ConvNext

        self.adapter = Adapter(512, 4).to(device)

        attr = []
        # now get the text features for all the gpt4 sentences
        for i in range(num_classes):
            # need to include code for all datasets, some dont need the folowing line
            current_sentences = torch.cat([tokenizer(c) for c in label_queue[i]])
            attention_masks = torch.where(current_sentences != 0, 1, 0).to(device)
            current_sentences = current_sentences.to(device)

            with torch.no_grad():
                current_text_features = self.encode_text(current_sentences,attention_masks)
                attr.append(current_text_features)
        self.attr = torch.stack(attr)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        #return self.visual.conv1.weight.dtype
        return self.visual.dwconv.weight.dtype

    def encode_image(self, image, mode):
        # return self.visual(image.type(self.dtype), mode)
        return self.visual(image)
    def encode_text(self, text,attention_masks):
        # ratio = 0.8
        # #text[B,256]
        # x = self.transformer(text,attention_mask=attention_masks).last_hidden_state
        # x = x[:,0,:]
        # x = self.text_proj(x)
        # x1 = self.adapter(x)
        # #x = torch.tensor(x).to(device)
        # x = ratio * x + (1-ratio) * x1
        # #从[8,77,512]到[8,512]
        ratio = 0.8
        x = self.biomedclip.text(text)
        x1 = self.adapter(x)
        x = ratio * x + (1 - ratio) * x1
        #直接编码成[B,512]
        return F.normalize(x, dim=-1)

    def forward(self, image,text,labels,attention_masks, attention_masks_parent=None,text_queue_1=None, text_queue_2=None,text_queue_3=None, text_queue_4=None):
        imgage_prob, image_features,attn_features = self.encode_image(image, mode='train')
        alpha_1=0.7
        loss1 = open_clip.ClipLoss()

        #caculate visual prompt


        if self.training:
            self.biomedclip.train()
            #print(text.shape)

            device = image.device
            text_features_1 = self.attr
           # print(f'text_featrue_1 shape {text_features_1.shape}')

            text_features_2 = self.adapter_self(text_features_1)
            text_features_1 = text_features_2.mean(dim=1)

            text_features = []
            text_features_3 = []
            for label in labels:
                text_features.append(text_features_1[int(label),:])
                text_features_3.append(self.attr[int(label),:])
            text_features = torch.stack(text_features)
            text_features_3 = torch.stack(text_features_3)

            local_textmixed_vision_feature = self.contextDecoder(text=text_features_3, visual=attn_features.permute(0,2, 3, 1),mixWho='visual')
            local_textmixed_vision_feature = local_textmixed_vision_feature.mean(dim=1)

            local_textmixed_text_feature = self.contextDecoder(text=text_features_3, visual=attn_features.permute(0,2, 3, 1),mixWho='text')
            local_textmixed_text_feature = local_textmixed_text_feature.mean(dim=1)
            
            
            #融合。
            combine_pram = 1
            image_features = image_features + combine_pram * local_textmixed_text_feature
            #text_features = text_features + combine_pram * local_textmixed_vision_feature
            # text_features = self.encode_text(text,attention_masks)
            # x = self.adapter(text_features).to(device)
            #
            # ratio = 0.2
            # text_features = ratio * x + (1 - ratio) * text_features
            # text_features = text_features.mean(dim=1)
            # if text_queue_1 != None:
            #     text_features_1 = self.encode_text(text_queue_1, attention_masks_parent)
            #     x1 = self.adapter(text_features_1).to(device)
            #     text_features_1 = ratio * x1 + (1 - ratio) * text_features_1
            #     text_features = alpha_1 * text_features + (1-alpha_1) * text_features_1
            text_features = text_features / text_features.norm(dim=-1,keepdim=True)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            logit_scale = self.logit_scale.exp()

            loss_clip = loss1(image_features, text_features,logit_scale)
            #loss_clip = clip_loss(logits_per_image, logits_per_text,device)

            return loss_clip, imgage_prob
        else:
            return torch.tensor(0).long(), imgage_prob
