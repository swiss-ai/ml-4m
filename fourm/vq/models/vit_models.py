# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import warnings
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange

# xFormers imports
try:
    from xformers.ops import memory_efficient_attention, unbind

    XFORMERS_AVAILABLE = True
except ImportError:
    print("xFormers not available")
    XFORMERS_AVAILABLE = False


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.0):
    """Sine-cosine positional embeddings as used in MoCo-v3"""
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
    assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    pos_emb = rearrange(pos_emb, "b (h w) d -> b d h w", h=h, w=w, d=embed_dim)
    return pos_emb


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        if XFORMERS_AVAILABLE:
            q, k, v = unbind(qkv, 2)  # Each is of shape B x N x num_heads x C // num_heads
            x = memory_efficient_attention(q, k, v)
            x = x.reshape([B, N, C])
        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, **kwargs):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.query_norm = norm_layer(dim)
        self.context_norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, context, **kwargs):
        x = x + self.drop_path(self.self_attn(self.norm1(x)))
        x = x + self.drop_path(self.cross_attn(self.query_norm(x), self.context_norm(context)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    From https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch.

    From https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ViTEncoder(nn.Module):
    """Transformer to map images / feature maps to latent features.

    Args:
        in_channels: Number of input channels.
        patch_size: Patch size.
        resolution: Image resolution.
        dim_tokens: Transformer dimension.
        depth: Number of transformer layers.
        num_heads: Number of attention heads.
        mlp_ratio: MLP ratio.
        qkv_bias: If True, add bias to the qkv projection.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalization layer.
        sincos_pos_emb: If True, use sine-cosine positional embedding.
        learnable_pos_emb: If True, learn positional embedding.
        patch_proj: If True, project image patches to tokens.
          Consider disabling when encoding feature maps.
        post_mlp: If True, add MLP after transformer.
          See https://arxiv.org/abs/2110.04627.
        ckpt_path: Path to checkpoint to load.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        patch_size: int = 16,
        resolution: int = 256,
        dim_tokens: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        sincos_pos_emb: bool = True,
        learnable_pos_emb: bool = False,
        patch_proj: bool = True,
        post_mlp: bool = False,
        ckpt_path: Optional[str] = None,
        **ignore_kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.P_H, self.P_W = pair(patch_size)
        self.H, self.W = pair(resolution)
        self.dim_tokens = dim_tokens
        self.patch_proj = patch_proj

        assert (self.H % self.P_H == 0) and (
            self.W % self.P_W == 0
        ), f"Image sizes {self.H}x{self.W} must be divisible by patch sizes {self.P_H}x{self.P_W}"

        N_H = self.H // self.P_H
        N_W = self.W // self.P_W

        if sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=N_H, w=N_W, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, N_H, N_W))
            trunc_normal_(self.pos_emb, std=0.02)

        # Image patches -> tokens projection
        if patch_proj:
            self.proj = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.dim_tokens,
                kernel_size=(self.P_H, self.P_W),
                stride=(self.P_H, self.P_W),
            )
        else:
            self.proj = nn.Conv2d(in_channels=self.in_channels, out_channels=self.dim_tokens, kernel_size=1, stride=1)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=dim_tokens,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        if post_mlp:
            self.norm_mlp = norm_layer(dim_tokens)
            self.post_mlp = Mlp(dim_tokens, int(mlp_ratio * dim_tokens), act_layer=nn.Tanh)

        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif "kv" in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv2d):
                if ".proj" in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if ckpt_path is not None:
            print(f"Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path)
            ckpt["model"]["pos_emb"] = rearrange(
                ckpt["model"]["pos_embed"][:, 1:], "b (nh nw) d -> b d nh nw", nh=N_H, nw=N_W
            )
            ckpt["model"]["proj.weight"] = ckpt["model"]["patch_embed.proj.weight"]
            ckpt["model"]["proj.bias"] = ckpt["model"]["patch_embed.proj.bias"]
            msg = self.load_state_dict(ckpt["model"], strict=False)
            print(msg)

    def _init_weights(self, m: nn.Module) -> None:
        """Weight initialization"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self) -> int:
        """Get number of transformer layers."""
        return len(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ViT encoder forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W] or
              [B, C, N_H, N_W] (patch projection disabled).

        Returns:
            Output tensor of shape [B, dim_tokens, N_H, N_W].
        """
        # Create patches [B, C, H, W] -> [B, (H*W), C]
        if self.patch_proj:
            B, C, H, W = x.shape
            assert (H % self.P_H == 0) and (
                W % self.P_W == 0
            ), f"Image sizes {H}x{W} must be divisible by patch sizes {self.P_H}x{self.P_W}"
            N_H, N_W = H // self.P_H, W // self.P_W  # Number of patches in height and width
        else:
            B, C, N_H, N_W = x.shape
        x = rearrange(self.proj(x), "b d nh nw -> b (nh nw) d")

        if self.pos_emb is not None:
            # Create positional embedding
            x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode="bicubic", align_corners=False)
            x_pos_emb = rearrange(x_pos_emb, "b d nh nw -> b (nh nw) d")
            # Add positional embeddings to patches
            x = x + x_pos_emb

        # Transformer forward pass
        x = self.blocks(x)

        if hasattr(self, "post_mlp"):
            with autocast(enabled=False):
                x = x.float() + self.post_mlp(self.norm_mlp(x.float()))

        # Reshape into 2D grid
        x = rearrange(x, "b (nh nw) d -> b d nh nw", nh=N_H, nw=N_W)

        return x


class ViTDecoder(nn.Module):
    """Transformer to map latent features back to images / feature maps.

    Args:
        out_channels: Number of output channels.
        patch_size: Patch size.
        resolution: Image resolution.
        dim_tokens: Transformer dimension.
        depth: Number of transformer layers.
        num_heads: Number of attention heads.
        mlp_ratio: MLP ratio.
        qkv_bias: If True, add bias to the qkv projection.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalization layer.
        sincos_pos_emb: If True, use sine-cosine positional embedding.
        learnable_pos_emb: If True, learn positional embedding.
        patch_proj: If True, reproject tokens back to images.
          Consider disabling when encoding feature maps.
        post_mlp: If True, add MLP before transformer.
          See https://arxiv.org/abs/2110.04627.
        out_conv: If True, add two ConvNeXt blocks after transformer
          to deal with patch checkerboard artifacts.
    """

    def __init__(
        self,
        *,
        out_channels: int = 3,
        patch_size: int = 16,
        resolution: int = 256,
        dim_tokens: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        sincos_pos_emb: bool = True,
        learnable_pos_emb: bool = False,
        patch_proj: bool = True,
        post_mlp: bool = False,
        out_conv: bool = False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.P_H, self.P_W = pair(patch_size)
        self.H, self.W = pair(resolution)
        self.dim_tokens = dim_tokens
        self.patch_proj = patch_proj

        assert (self.H % self.P_H == 0) and (
            self.W % self.P_W == 0
        ), f"Image sizes {self.H}x{self.W} must be divisible by patch sizes {self.P_H}x{self.P_W}"

        N_H = self.H // self.P_H
        N_W = self.W // self.P_W

        if sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=N_H, w=N_W, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, N_H, N_W))
            trunc_normal_(self.pos_emb, std=0.02)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=dim_tokens,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Tokens -> image output projection
        if post_mlp:
            self.norm_mlp = norm_layer(dim_tokens)
            self.post_mlp = Mlp(dim_tokens, int(mlp_ratio * dim_tokens), act_layer=nn.Tanh)
        if patch_proj:
            self.out_proj = nn.Linear(dim_tokens, self.out_channels * self.P_H * self.P_W)
        else:
            self.out_proj = nn.Linear(dim_tokens, self.out_channels)
        if out_conv:
            self.out_conv = nn.Sequential(ConvNeXtBlock(dim=self.out_channels), ConvNeXtBlock(dim=self.out_channels))

        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif "kv" in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6.0 / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv2d):
                if ".proj" in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m: nn.Module) -> None:
        """Weight initialization"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self) -> int:
        """Get number of transformer layers."""
        return len(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ViT decoder forward pass.

        Args:
            x: Input tensor of shape [B, dim_tokens, N_H, N_W].

        Returns:
            Output tensor of shape [B, C, H, W] or
              [B, C, N_H, N_W] (patch projection disabled).
        """
        B, D, N_H, N_W = x.shape

        # Reshape into 1D
        x = rearrange(x, "b d nh nw -> b (nh nw) d")

        if self.pos_emb is not None:
            # Create positional embedding
            x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode="bicubic", align_corners=False)
            x_pos_emb = rearrange(x_pos_emb, "b d nh nw -> b (nh nw) d")
            # Add positional embeddings to patches
            x = x + x_pos_emb

        # Transformer forward pass
        x = self.blocks(x)

        # Project each token to (C * P_H * P_W)
        if hasattr(self, "post_mlp"):
            x = x + self.post_mlp(self.norm_mlp(x))
        x = self.out_proj(x)

        # Reshape sequence of patches into image or output features
        ph, pw = (self.P_H, self.P_W) if self.patch_proj else (1, 1)
        x = rearrange(
            x, "b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)", nh=N_H, nw=N_W, ph=ph, pw=pw, c=self.out_channels
        )

        # Optional conv layers to reduce patch artifacts
        if hasattr(self, "out_conv"):
            x = self.out_conv(x)

        return x


# Encoder presets


def vit_s_enc(
    in_channels,
    patch_size,
    resolution,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    sincos_pos_emb=True,
    learnable_pos_emb=False,
    patch_proj=True,
    post_mlp=False,
):
    model = ViTEncoder(
        in_channels=in_channels,
        patch_size=patch_size,
        resolution=resolution,
        dim_tokens=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        sincos_pos_emb=sincos_pos_emb,
        learnable_pos_emb=learnable_pos_emb,
        patch_proj=patch_proj,
        post_mlp=post_mlp,
    )
    return model


def vit_b_enc(
    in_channels,
    patch_size,
    resolution,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    sincos_pos_emb=True,
    learnable_pos_emb=False,
    patch_proj=True,
    post_mlp=False,
    ckpt_path=None,
):
    model = ViTEncoder(
        in_channels=in_channels,
        patch_size=patch_size,
        resolution=resolution,
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        sincos_pos_emb=sincos_pos_emb,
        learnable_pos_emb=learnable_pos_emb,
        patch_proj=patch_proj,
        post_mlp=post_mlp,
        ckpt_path=ckpt_path,
    )
    return model


def vit_l_enc(
    in_channels,
    patch_size,
    resolution,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    sincos_pos_emb=True,
    learnable_pos_emb=False,
    patch_proj=True,
    post_mlp=False,
):
    model = ViTEncoder(
        in_channels=in_channels,
        patch_size=patch_size,
        resolution=resolution,
        dim_tokens=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        sincos_pos_emb=sincos_pos_emb,
        learnable_pos_emb=learnable_pos_emb,
        patch_proj=patch_proj,
        post_mlp=post_mlp,
    )
    return model


# Decoder presets


def vit_s_dec(
    out_channels,
    patch_size,
    resolution,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    sincos_pos_emb=True,
    learnable_pos_emb=False,
    patch_proj=True,
    post_mlp=False,
    out_conv=False,
):
    model = ViTDecoder(
        out_channels=out_channels,
        patch_size=patch_size,
        resolution=resolution,
        dim_tokens=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        sincos_pos_emb=sincos_pos_emb,
        learnable_pos_emb=learnable_pos_emb,
        patch_proj=patch_proj,
        post_mlp=post_mlp,
        out_conv=out_conv,
    )
    return model


def vit_b_dec(
    out_channels,
    patch_size,
    resolution,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    sincos_pos_emb=True,
    learnable_pos_emb=False,
    patch_proj=True,
    post_mlp=False,
    out_conv=False,
):
    model = ViTDecoder(
        out_channels=out_channels,
        patch_size=patch_size,
        resolution=resolution,
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        sincos_pos_emb=sincos_pos_emb,
        learnable_pos_emb=learnable_pos_emb,
        patch_proj=patch_proj,
        post_mlp=post_mlp,
        out_conv=out_conv,
    )
    return model


def vit_l_dec(
    out_channels,
    patch_size,
    resolution,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    sincos_pos_emb=True,
    learnable_pos_emb=False,
    patch_proj=True,
    post_mlp=False,
    out_conv=False,
):
    model = ViTDecoder(
        out_channels=out_channels,
        patch_size=patch_size,
        resolution=resolution,
        dim_tokens=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        sincos_pos_emb=sincos_pos_emb,
        learnable_pos_emb=learnable_pos_emb,
        patch_proj=patch_proj,
        post_mlp=post_mlp,
        out_conv=out_conv,
    )
    return model
