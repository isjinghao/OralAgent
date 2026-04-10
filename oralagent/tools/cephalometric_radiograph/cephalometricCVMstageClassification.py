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


############################### for Dinov3 Classification Model ##################################
import json
import os
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from oralagent.tools.model_DINOv3.dinov3_classifier import DinoV3Classifier


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class CephalometricImageCVMstagesClassificationInput(BaseModel):
    """Input schema for the Cephalometric Image CVM Status Classification Tool."""

    image_path: str = Field(..., description="Path to the Cephalometric image file to be processed")

class CephalometricImageCVMstagesClassificationOutput(BaseModel):
    """Output schema for the Cephalometric Image CVM Status Classification Tool."""

    predicted_class: str = Field(..., description="Predicted Cephalometric Image CVM Status class")
    confidence: float = Field(..., description="Prediction confidence (0-1)")


class CephalometricImageCVMstagesClassificationTool(BaseTool):
    """Tool for performing image-level CVM status analysis of cephmetric images."""

    name: str = "cephalometric_xray_cvm_stages_classification"
    description: str = (
        "Classifies cephmetric images into 6 stages related to image-level abnormalities, including CVM-S1, CVM-S2, CVM-S3, CVM-S4, CVM-S5, CVM-S6."
        "Ensure the input cephmetric image is of high resolution and quality for accurate classification."
    )
    
    args_schema: Type[BaseModel] = CephalometricImageCVMstagesClassificationInput

    checkpoint_path: str = ""
    coco_names_path: str = ""
    device: Optional[str] = "cuda"
    transform: Any = None
    temp_dir: Path = Path("temp")
    model: Any = None
    id2name: Any = None

    def __init__(self, checkpoint_path: str, coco_names_path: str, device: Optional[str] = "cuda", temp_dir: Optional[Path] = Path("temp")):
        """Initialize the Cephalometric Image CVM Status Classification Tool."""
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.coco_names_path = coco_names_path
        self.device = torch.device(device) if device else "cuda"
        self.id2name = self._load_category_names()
        self.model = self._load_model(self.checkpoint_path, len(self.id2name))
        
        self.temp_dir = temp_dir if isinstance(temp_dir, Path) else Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def _load_model(self, checkpoint_path: str, num_classes: int):
        """Load the complete Cephalometric Image CVM Status classifier model."""
        # Create model instance
        model = DinoV3Classifier(task_name="cephalometric_xray_cvm_stages_classification", num_classes=num_classes)
        
        state_dict = load_file(checkpoint_path)
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"Weights {checkpoint_path.split('/')[-1]} loaded successfully (strict mode).")
        except RuntimeError as e:
            raise RuntimeError(f"Error loading model weights: {e}. Check if the checkpoint is compatible with the model architecture.")
        
        model.to(self.device)
        model.eval()
        
        return model

    def _load_category_names(self):
        """Load category names from COCO format."""
        with open(self.coco_names_path, 'r') as f:
            categories = json.load(f)
        return {int(cat_id): cat_name for cat_name, cat_id in categories.items()}

    def _read_image(self, img_path, aug):
        image = Image.open(img_path).convert("RGB")
        raw_shape = image.size[::-1]
        input_tensor = aug(image).unsqueeze(0)
        return input_tensor, raw_shape
    
    def _preprocess_image(self, image_path: str):
            """Preprocess the input image."""
            transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image, raw_shape = self._read_image(image_path, transform)
            return image, raw_shape

    def _run_inference(self, image_tensor, raw_shape):
        """Run inference on the input image tensor."""
        self.model.to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor.to(self.device))

        probs = torch.softmax(features, dim=1)
        pred_conf, pred_class = torch.max(probs, dim=1)

        return pred_class.cpu().numpy(), pred_conf.cpu().numpy()

    def _run(self, 
            image_path: str,
            # run_manager: Optional[CallbackManagerForToolRun] = None,
            ) -> CephalometricImageCVMstagesClassificationOutput:
        try:
            """Run the Cephalometric Image CVM Status Classification Tool."""

            # Preprocess the image
            image_tensor, orig_size = self._preprocess_image(image_path)
            # Run inference
            pred_class, pred_conf = self._run_inference(image_tensor, orig_size)

            # Create output object
            output = CephalometricImageCVMstagesClassificationOutput(predicted_class=self.id2name[pred_class[0]], confidence=float(pred_conf[0]))
            # Save visualization
            viz_path = None
            # viz_path = self._save_visualization(
            #         image_path=image_path,
            #         pred_class=self.id2name[pred_class[0]],
            #         )

            # Prepare output and metadata
            output = {
                "detection_image_path": viz_path,
                "pred_class": self.id2name[pred_class[0]],
                "confidence": float(pred_conf[0])
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
            # Handle errors and prepare error output and metadata
            # raise KeyError(f"Error during tool execution: {e}")
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
        landmark_names: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_path, landmark_names)

    def _save_visualization(self, image_path: str, pred_class: str) -> str:
        """
        Save a visualization of the predictions for cephalometric CVM status classification.
        """
        # Load the original image
        orig_image = Image.open(image_path).convert("RGB")

        # Create a figure and axis for visualization
        fig, ax = plt.subplots(1, figsize=(12, 10))
        ax.imshow(orig_image)
        ax.axis('off')
        ax.set_title(f"Cephalometric CVM Status {os.path.basename(image_path)} classification results: {pred_class}", fontsize=16)
        plt.axis('off')
        # Save the visualization
        save_path = self.temp_dir / f"Cephalometric_CVM_Status_Classification_{uuid.uuid4().hex[:8]}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0) 
        plt.close()
        print(f"Visualization saved to: {save_path}")
        return str(save_path)