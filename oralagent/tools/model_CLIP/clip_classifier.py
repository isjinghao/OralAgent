import json
import torch
import torch.nn as nn
from transformers import CLIPVisionModel

class BioMedCLIPClassifier(nn.Module):
    def __init__(self, checkpoint_path, coco_names_path, num_classes, device="cuda"):
        super(BioMedCLIPClassifier, self).__init__()
        
        self.device = device
        self.coco_names_path = coco_names_path
        self.id2name = self._load_category_names()
        
        print(f"Loading BioMedCLIP Vision Encoder from chuhac/BiomedCLIP-vit-bert-hf...")
        self.backbone = CLIPVisionModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf")
        self.hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_classes)

        # 加载权重
        self._load_weights(checkpoint_path)
        
        self.to(self.device)
        self.eval()

    def _load_weights(self, checkpoint_path: str):
        """加载 pth/safetensors 权重到当前模型"""
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # 如果权重文件包含了完整模型（有 backbone + classifier）
        try:
            self.load_state_dict(state_dict, strict=True)
            print("Weights loaded successfully (strict mode).")
        except RuntimeError as e:
            # 如果 strict 失败，试试只加载匹配的 key
            print(f"Strict loading failed: {e}")
            print("Trying non-strict loading...")
            self.load_state_dict(state_dict, strict=False)
            print("Weights loaded (non-strict mode).")

    def _load_category_names(self):
        """Load category names from COCO format."""
        with open(self.coco_names_path, 'r') as f:
            categories = json.load(f)
        return {int(cat_id): cat_name for cat_name, cat_id in categories.items()}


    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        features = outputs.pooler_output
        
        logits = self.classifier(features)
        return logits

    def predict(self, image_path: str) -> dict:
        """供 Agent 预处理调用"""
        from transformers import CLIPImageProcessor
        from PIL import Image
        
        processor = CLIPImageProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf")
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            logits = self.forward(inputs["pixel_values"])
            probs = torch.softmax(logits, dim=-1)
            conf, pred_id = probs.max(dim=-1)
        
        return {
            "modality": self.id2name[pred_id.item()],
            "confidence": conf.item()
        }
