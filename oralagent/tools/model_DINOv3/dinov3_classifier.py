import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

# 与当前 py 文件同目录下的 config 文件夹（仅结构，不加载官方权重）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.join(_SCRIPT_DIR, "configs_dinov3-convnext-base-pretrain-lvd1689m")


class DinoV3Classifier(nn.Module):
    """Complete DinoV3 classifier model."""
    
    def __init__(self, task_name, num_classes, hidden_size=1024, dropout=0.4):
        super().__init__()
        
        # 只从本地 config 构建 backbone 结构，不加载权重（避免 HF repo id 校验）
        config = AutoConfig.from_pretrained(_CONFIG_DIR, local_files_only=True)
        self.backbone = AutoModel.from_config(config)
        
        if task_name == "cephalometric_xray_cvm_stages_classification":
            self.head = nn.Sequential(
                nn.BatchNorm1d(hidden_size, affine=True),
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            )
        else:
            self.head = nn.Sequential(
                    nn.BatchNorm1d(hidden_size, affine=True),
                    nn.Linear(hidden_size, num_classes)
            )
    
    def forward(self, pixel_values):
        # Extract features from backbone
        outputs = self.backbone(pixel_values)
        
        last_hidden = outputs.pooler_output
        last_hidden = self.head(last_hidden)
        logits = last_hidden
        return logits