from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid

import numpy as np
import torch
import matplotlib.pyplot as plt
import traceback

from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


############################### for ViT Classification Model ##################################
import json
import os
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from oralagent.tools.model_ViT.vit_image_classification import ViTClassifier


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class PeriapicalXRayDisease7ClassificationInput(BaseModel):
    """Input schema for the Periapical X-Ray Disease 7 Classification Tool."""

    image_path: str = Field(..., description="Path to the Periapical X-Ray image file to be processed")


class PeriapicalXRayDisease7ClassificationOutput(BaseModel):
    """Output schema for the Periapical X-Ray Disease 7 Classification Tool."""

    predicted_class: str = Field(..., description="Predicted Disease class")
    confidence: float = Field(..., description="Prediction confidence (0-1)")


class PeriapicalXRayDisease7ClassificationTool(BaseTool):
    """Tool for performing Periapical X-Ray Disease 7 classification using ViT."""
    
    name: str = "periapical_xray_disease_7_classification"
    description: str = (
        "Classifies periapical x-ray into 7 disease categories including: "
        "Irreversible Pulpitis, Impacted Tooth, Apical Periodontitis, Bone loss, Mixed Tooth Types, Caries, Periodontitis. "
        "Ensure the input periapical x-ray image is of high resolution and quality for accurate classification."
    )

    args_schema: Type[BaseModel] = PeriapicalXRayDisease7ClassificationInput

    checkpoint_path: str = ""
    coco_names_path: str = ""

    device: Optional[str] = "cuda"
    temp_dir: Path = Path("temp")

    model: Any = None
    img_size: int = 512
    id2name: Any = None

    def __init__(
        self,
        checkpoint_path: str,
        coco_names_path: str,
        device: Optional[str] = "cuda",
        temp_dir: Optional[Path] = Path("temp"),
        model_type: str = "ViT-L_16",
        img_size: int = 512,
    ):
        """Initialize the Periapical X-Ray Disease 7 Classification Tool (ViT)."""
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.coco_names_path = coco_names_path
        self.device = torch.device(device) if device else torch.device("cuda")
        self.img_size = img_size
        self.id2name = self._load_category_names()
        # self.model = self._load_model(self.checkpoint_path, len(self.id2name), model_type)
        self.model = self._load_model(self.checkpoint_path, 100, model_type)

        self.temp_dir = temp_dir if isinstance(temp_dir, Path) else Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def _remap_vit_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remap checkpoint keys from _VisionTransformer format to ViTClassifier format.
        Checkpoint saved from _VisionTransformer has keys like 'transformer.xxx';
        ViTClassifier expects 'model.transformer.xxx'.
        """
        if not state_dict:
            return state_dict
        first_key = next(iter(state_dict.keys()))
        if first_key.startswith("model."):
            return state_dict
        return {"model." + k: v for k, v in state_dict.items()}

    def _load_model(self, checkpoint_path: str, num_classes: int, model_type: str):
        """Load the ViT-based Disease 7 Classification model."""
        model = ViTClassifier(
            task_name="periapical_xray_disease_7class",
            num_classes=num_classes,
            model_type=model_type,
            img_size=self.img_size,
        )

        if checkpoint_path.lower().endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        # Remap checkpoint keys: checkpoint may be saved from _VisionTransformer (keys like
        # "transformer.embeddings.xxx") but ViTClassifier expects "model.transformer.xxx".
        state_dict = self._remap_vit_state_dict(state_dict)

        try:
            model.load_state_dict(state_dict, strict=True)
            print("ViT weights loaded successfully (strict mode).")
        except RuntimeError as e:
            raise RuntimeError(
                f"Error loading model weights: {e}. Check if the checkpoint is compatible with the model architecture."
            )

        model.to(self.device)
        model.eval()
        return model

    def _load_category_names(self):
        """Load category names from COCO format."""
        with open(self.coco_names_path, "r") as f:
            categories = json.load(f)
        return {int(cat_id): cat_name for cat_name, cat_id in categories.items()}

    def _read_image(self, img_path, aug):
        image = Image.open(img_path).convert("RGB")
        raw_shape = image.size[::-1]
        input_tensor = aug(image).unsqueeze(0)
        return input_tensor, raw_shape

    def _preprocess_image(self, image_path: str):
        """Preprocess the input image for ViT (ImageNet norm, resize to img_size)."""
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image, raw_shape = self._read_image(image_path, transform)
        return image, raw_shape

    def _run_inference(self, image_tensor, raw_shape):
        """Run inference on the input image tensor."""
        self.model.to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor.to(self.device))

        probs = torch.softmax(logits, dim=1)
        pred_conf, pred_class = torch.max(probs, dim=1)
        return pred_class.cpu().numpy(), pred_conf.cpu().numpy()

    def _run(
        self,
        image_path: str,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Run the Periapical X-Ray Disease 7 Classification Tool."""
        try:
            image_tensor, orig_size = self._preprocess_image(image_path)
            pred_class, pred_conf = self._run_inference(image_tensor, orig_size)

            pred_name = self.id2name.get(int(pred_class[0]), f"class_{pred_class[0]}")
            conf = float(pred_conf[0])

            viz_path = None
            # viz_path = self._save_visualization(
            #         image_path=image_path,
            #         pred_class=self.id2name[pred_class[0]],
            #         )

            output = {
                "detection_image_path": viz_path,
                "pred_class": pred_name,
                "confidence": conf,
            }

            metadata = {
                "image_path": image_path,
                "detection_image_path": viz_path,
                "original_size": orig_size,
                "model_size": tuple(image_tensor.shape[-2:]),
                "analysis_status": "completed",
            }
            return output, metadata

        except Exception as e:
            error_output = {"error": str(e)}
            error_metadata = {
                "image_path": image_path,
                "analysis_status": "failed",
                "error_traceback": traceback.format_exc(),
            }
            return error_output, error_metadata

    async def _arun(
        self,
        image_path: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_path)

    def _save_visualization(self, image_path: str, pred_class: str) -> str:
        """Save a visualization of the predictions for periapical disease classification."""
        orig_image = Image.open(image_path).convert("RGB")
        fig, ax = plt.subplots(1, figsize=(12, 10))
        ax.imshow(orig_image)
        ax.axis("off")
        ax.set_title(
            f"Periapical Disease {os.path.basename(image_path)} classification results: {pred_class}",
            fontsize=16,
        )
        plt.axis("off")
        save_path = self.temp_dir / f"periapical_disease_7classification_{uuid.uuid4().hex[:8]}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Visualization saved to: {save_path}")
        return str(save_path)
