# coding=utf-8
"""
ViT 分类模型定义与推理接口，全部逻辑自包含，不依赖外部 ViT 项目。
与 dinov3_classifier 一致：仅实现模型定义及 forward 推理，权重由外部加载。
"""
from __future__ import absolute_import, division, print_function

import copy
import math
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage


# ---------- 配置（替代 ml_collections） ----------
class _ConfigDict:
    """简单 ConfigDict，支持 .attr 与 .get(key)。"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)


def _get_testing():
    c = _ConfigDict(
        patches=_ConfigDict(size=(16, 16)),
        hidden_size=1,
        transformer=_ConfigDict(mlp_dim=1, num_heads=1, num_layers=1, attention_dropout_rate=0.0, dropout_rate=0.1),
        classifier="token",
        representation_size=None,
    )
    return c


def _get_b16_config():
    c = _ConfigDict(
        patches=_ConfigDict(size=(16, 16)),
        hidden_size=768,
        transformer=_ConfigDict(
            mlp_dim=3072,
            num_heads=12,
            num_layers=12,
            attention_dropout_rate=0.0,
            dropout_rate=0.1,
        ),
        classifier="token",
        representation_size=None,
    )
    return c


def _get_b32_config():
    c = _get_b16_config()
    c.patches = _ConfigDict(size=(32, 32))
    return c


def _get_l16_config():
    c = _ConfigDict(
        patches=_ConfigDict(size=(16, 16)),
        hidden_size=1024,
        transformer=_ConfigDict(
            mlp_dim=4096,
            num_heads=16,
            num_layers=24,
            attention_dropout_rate=0.0,
            dropout_rate=0.1,
        ),
        classifier="token",
        representation_size=None,
    )
    return c


def _get_l32_config():
    c = _get_l16_config()
    c.patches = _ConfigDict(size=(32, 32))
    return c


def _get_h14_config():
    c = _ConfigDict(
        patches=_ConfigDict(size=(14, 14)),
        hidden_size=1280,
        transformer=_ConfigDict(
            mlp_dim=5120,
            num_heads=16,
            num_layers=32,
            attention_dropout_rate=0.0,
            dropout_rate=0.1,
        ),
        classifier="token",
        representation_size=None,
    )
    return c


def _get_r50_b16_config():
    c = _get_b16_config()
    c.patches = _ConfigDict(grid=(14, 14))
    c.resnet = _ConfigDict(num_layers=(3, 4, 9), width_factor=1)
    return c


# ---------- ResNet V2（R50-ViT 用） ----------
def _np2th(weights, conv=False):
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class _StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return torch.nn.functional.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def _conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return _StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def _conv1x1(cin, cout, stride=1, bias=False):
    return _StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


class _PreActBottleneck(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = _conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = _conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = _conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or cin != cout:
            self.downsample = _conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        self.conv1.weight.copy_(_np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True))
        self.conv2.weight.copy_(_np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True))
        self.conv3.weight.copy_(_np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True))
        self.gn1.weight.copy_(_np2th(weights[pjoin(n_block, n_unit, "gn1/scale")]).view(-1))
        self.gn1.bias.copy_(_np2th(weights[pjoin(n_block, n_unit, "gn1/bias")]).view(-1))
        self.gn2.weight.copy_(_np2th(weights[pjoin(n_block, n_unit, "gn2/scale")]).view(-1))
        self.gn2.bias.copy_(_np2th(weights[pjoin(n_block, n_unit, "gn2/bias")]).view(-1))
        self.gn3.weight.copy_(_np2th(weights[pjoin(n_block, n_unit, "gn3/scale")]).view(-1))
        self.gn3.bias.copy_(_np2th(weights[pjoin(n_block, n_unit, "gn3/bias")]).view(-1))
        if hasattr(self, "downsample"):
            self.downsample.weight.copy_(
                _np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            )
            self.gn_proj.weight.copy_(
                _np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")]).view(-1)
            )
            self.gn_proj.bias.copy_(
                _np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")]).view(-1)
            )


class _ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        from collections import OrderedDict

        width = int(64 * width_factor)
        self.width = width
        self.root = nn.Sequential(
            OrderedDict([
                ("conv", _StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
                ("gn", nn.GroupNorm(32, width, eps=1e-6)),
                ("relu", nn.ReLU(inplace=True)),
                ("pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            ])
        )
        self.body = nn.Sequential(
            OrderedDict([
                (
                    "block1",
                    nn.Sequential(
                        OrderedDict(
                            [("unit1", _PreActBottleneck(cin=width, cout=width * 4, cmid=width))]
                            + [
                                (
                                    f"unit{i}",
                                    _PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width),
                                )
                                for i in range(2, block_units[0] + 1)
                            ]
                        )
                    ),
                ),
                (
                    "block2",
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "unit1",
                                    _PreActBottleneck(
                                        cin=width * 4,
                                        cout=width * 8,
                                        cmid=width * 2,
                                        stride=2,
                                    ),
                                )
                            ]
                            + [
                                (
                                    f"unit{i}",
                                    _PreActBottleneck(
                                        cin=width * 8, cout=width * 8, cmid=width * 2
                                    ),
                                )
                                for i in range(2, block_units[1] + 1)
                            ]
                        )
                    ),
                ),
                (
                    "block3",
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "unit1",
                                    _PreActBottleneck(
                                        cin=width * 8,
                                        cout=width * 16,
                                        cmid=width * 4,
                                        stride=2,
                                    ),
                                )
                            ]
                            + [
                                (
                                    f"unit{i}",
                                    _PreActBottleneck(
                                        cin=width * 16, cout=width * 16, cmid=width * 4
                                    ),
                                )
                                for i in range(2, block_units[2] + 1)
                            ]
                        )
                    ),
                ),
            ])
        )

    def forward(self, x):
        x = self.root(x)
        x = self.body(x)
        return x


# ---------- ViT 组件 ----------
def _swish(x):
    return x * torch.sigmoid(x)


_ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": _swish}

_ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
_ATTENTION_K = "MultiHeadDotProductAttention_1/key"
_ATTENTION_V = "MultiHeadDotProductAttention_1/value"
_ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
_FC_0 = "MlpBlock_3/Dense_0"
_FC_1 = "MlpBlock_3/Dense_1"
_ATTENTION_NORM = "LayerNorm_0"
_MLP_NORM = "LayerNorm_2"


class _Attention(nn.Module):
    def __init__(self, config, vis):
        super(_Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class _Mlp(nn.Module):
    def __init__(self, config):
        super(_Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = _ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class _Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(_Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (
                img_size[0] // 16 // grid_size[0],
                img_size[1] // 16 // grid_size[1],
            )
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False
        if self.hybrid:
            self.hybrid_model = _ResNetV2(
                block_units=config.resnet.num_layers,
                width_factor=config.resnet.width_factor,
            )
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class _Block(nn.Module):
    def __init__(self, config, vis):
        super(_Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = _Mlp(config)
        self.attn = _Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = _np2th(weights[pjoin(ROOT, _ATTENTION_Q, "kernel")]).view(
                self.hidden_size, self.hidden_size
            ).t()
            key_weight = _np2th(weights[pjoin(ROOT, _ATTENTION_K, "kernel")]).view(
                self.hidden_size, self.hidden_size
            ).t()
            value_weight = _np2th(weights[pjoin(ROOT, _ATTENTION_V, "kernel")]).view(
                self.hidden_size, self.hidden_size
            ).t()
            out_weight = _np2th(weights[pjoin(ROOT, _ATTENTION_OUT, "kernel")]).view(
                self.hidden_size, self.hidden_size
            ).t()
            query_bias = _np2th(weights[pjoin(ROOT, _ATTENTION_Q, "bias")]).view(-1)
            key_bias = _np2th(weights[pjoin(ROOT, _ATTENTION_K, "bias")]).view(-1)
            value_bias = _np2th(weights[pjoin(ROOT, _ATTENTION_V, "bias")]).view(-1)
            out_bias = _np2th(weights[pjoin(ROOT, _ATTENTION_OUT, "bias")]).view(-1)
            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)
            mlp_weight_0 = _np2th(weights[pjoin(ROOT, _FC_0, "kernel")]).t()
            mlp_weight_1 = _np2th(weights[pjoin(ROOT, _FC_1, "kernel")]).t()
            mlp_bias_0 = _np2th(weights[pjoin(ROOT, _FC_0, "bias")]).t()
            mlp_bias_1 = _np2th(weights[pjoin(ROOT, _FC_1, "bias")]).t()
            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)
            self.attention_norm.weight.copy_(
                _np2th(weights[pjoin(ROOT, _ATTENTION_NORM, "scale")])
            )
            self.attention_norm.bias.copy_(
                _np2th(weights[pjoin(ROOT, _ATTENTION_NORM, "bias")])
            )
            self.ffn_norm.weight.copy_(_np2th(weights[pjoin(ROOT, _MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(_np2th(weights[pjoin(ROOT, _MLP_NORM, "bias")]))


class _Encoder(nn.Module):
    def __init__(self, config, vis):
        super(_Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            self.layer.append(copy.deepcopy(_Block(config, vis)))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class _Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(_Transformer, self).__init__()
        self.embeddings = _Embeddings(config, img_size=img_size)
        self.encoder = _Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class _VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(_VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = _Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(_np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(_np2th(weights["head/bias"]).t())
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                _np2th(weights["embedding/kernel"], conv=True)
            )
            self.transformer.embeddings.patch_embeddings.bias.copy_(
                _np2th(weights["embedding/bias"])
            )
            self.transformer.embeddings.cls_token.copy_(_np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(
                _np2th(weights["Transformer/encoder_norm/scale"])
            )
            self.transformer.encoder.encoder_norm.bias.copy_(
                _np2th(weights["Transformer/encoder_norm/bias"])
            )
            posemb = _np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)
                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(_np2th(posemb))
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)
            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    _np2th(weights["conv_root/kernel"], conv=True)
                )
                gn_weight = _np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = _np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


# 导出给 ViTClassifier 用的 CONFIGS
CONFIGS = {
    "ViT-B_16": _get_b16_config(),
    "ViT-B_32": _get_b32_config(),
    "ViT-L_16": _get_l16_config(),
    "ViT-L_32": _get_l32_config(),
    "ViT-H_14": _get_h14_config(),
    "R50-ViT-B_16": _get_r50_b16_config(),
    "testing": _get_testing(),
}


# ---------- 对外接口：ViTClassifier ----------
class ViTClassifier(nn.Module):
    """
    ViT 分类模型封装，接口与 DinoV3Classifier 对齐：
    - __init__(task_name, num_classes, model_type=..., img_size=...)
    - forward(pixel_values) -> logits
    权重不在此处加载，由调用方 load_state_dict / 加载 checkpoint。
    """

    def __init__(
        self,
        task_name,
        num_classes,
        model_type="ViT-B_16",
        img_size=224,
        zero_head=False,
    ):
        super().__init__()
        if model_type not in CONFIGS:
            raise ValueError(
                "model_type must be one of %s, got %s" % (list(CONFIGS.keys()), model_type)
            )
        config = CONFIGS[model_type]
        self.model = _VisionTransformer(
            config,
            img_size=img_size,
            num_classes=num_classes,
            zero_head=zero_head,
            vis=False,
        )
        self.num_classes = num_classes
        self.img_size = img_size
        self.model_type = model_type

    def forward(self, pixel_values):
        """推理：输入 [B, C, H, W]，返回 logits [B, num_classes]。"""
        logits, _ = self.model(pixel_values)
        return logits
