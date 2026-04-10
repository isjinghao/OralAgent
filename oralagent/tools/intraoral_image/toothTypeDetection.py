from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import traceback

from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool

############################### for DINO Detection Model ##################################
from ..util.transforms import *
from ..util import box_ops
from ..util.slconfig import SLConfig
from PIL import Image
import matplotlib.patches as patches
import os
import json


def build_model_main(args):
    from ..model_DINO.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

class IntraoralImageToothTypeDetectionInput(BaseModel):
    """Input schema for the Intraoral Image Dental Morphology Detection Tool."""
    image_path: str = Field(..., description="Path to the intraoral image file to be processed for dental morphology detection")
    dental_morphology_types: Optional[List[str]] = Field(
        None,
        description="A list of dental morphology type names to detect. If set to None, all available types will be detected. "
        "The available morphology types include: tooth, 1st Molar, 1st Premolar, 2nd Molar, 2nd Premolar, Canine, Central Incisor, Lateral Incisor "
    )

class IntraoralImageToothTypeDetectionOutput(BaseModel):
    """Output schema for Intraoral Image Dental Morphology Detection Tool."""
    detections: List[Dict[str, Any]] = Field(..., description="List of detected dental morphology regions and bounding boxes")


class IntraoralImageToothTypeDetectionTool(BaseTool):
    """Tool for performing dental morphology detection analysis on intraoral images."""

    name: str = "intraoral_image_teeth_number_and_type_detection"
    description: str = (
        "Detects the number and types of all teeth in intraoral images. "
        "It identifies eight tooth types: tooth, 1st Molar, 1st Premolar, 2nd Molar, 2nd Premolar, Canine, Central Incisor, Lateral Incisor"
        "Returns detection visualization and a list of detected tooth type regions with their bounding boxes and confidence scores. "
        "Ensure the input intraoral image is of high quality for optimal performance."
    )

    args_schema: Type[BaseModel] = IntraoralImageToothTypeDetectionInput

    config_path: str = ""
    checkpoint_path: str = ""
    coco_names_path: str = ""
    device: Optional[str] = "cuda"
    transform: Any = None
    temp_dir: Path = Path("temp")
    model: Any = None
    postprocessors: Any = None
    id2name: Any = None
    confidence: float = 0.5

    def __init__(self, config_path: str, checkpoint_path: str, coco_names_path: str, device: Optional[str] = "cuda", temp_dir: Optional[Path] = Path("temp")):
        """Initialize the Intraoral Image Dental Morphology Detection Tool."""
        super().__init__()
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.coco_names_path = coco_names_path
        self.device = torch.device(device) if device else "cuda"
        self.model, self.postprocessors = self._load_model()
        self.id2name = self._load_category_names()
        self.temp_dir = temp_dir if isinstance(temp_dir, Path) else Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.confidence = 0.5

    def _load_model(self):
        """Load the DINO model."""
        args = SLConfig.fromfile(self.config_path)
        args.device = self.device
        model, _, postprocessors = build_model_main(args)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=True)
        print(f"Model loaded successfully from {self.checkpoint_path}")
        model = model.to(self.device)
        model.eval()
        return model, postprocessors

    def _load_category_names(self):
        """Load category names from COCO format."""
        with open(self.coco_names_path, 'r') as f:
            categories = json.load(f)
        return {int(cat_id): cat_name for cat_id, cat_name in categories.items()}

    def _preprocess_image(self, image_path: str):
        """Preprocess the input image."""
        transform = Compose([
            RandomResize([800], max_size=1333),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size  # (w, h)
        image_transformed, _ = transform(image, None)
        return image_transformed, orig_size

    def _run_inference(self, image_tensor, orig_size):
        """Run inference on the input image tensor.
        
        Args:
            image_tensor: preprocessed image tensor
            orig_size: (orig_w, orig_h) tuple from PIL
        """
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            outputs = self.model(image_tensor)
            orig_w, orig_h = orig_size
            target_sizes = torch.tensor([[orig_h, orig_w]]).to(self.device)
            processed_outputs = self.postprocessors['bbox'](outputs, target_sizes)[0]
        return processed_outputs

    def _run(self, 
            image_path: str,
            dental_morphology_types: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None,
            ) -> IntraoralImageToothTypeDetectionOutput:
        try:
            # Preprocess the image
            image_tensor, orig_size = self._preprocess_image(image_path)
            outputs = self._run_inference(image_tensor, orig_size)

            # Process detections
            detections = []
            for i in range(len(outputs['boxes'])):
                score = outputs['scores'][i].item()

                if score < self.confidence:
                    continue

                box_xyxy = outputs['boxes'][i].cpu().numpy().tolist()
                x1, y1, x2, y2 = [round(c) for c in box_xyxy]

                label = outputs['labels'][i].item()
                morphology_name = self.id2name.get(label, f"Unknown ({label})")

                detections.append({
                    "dental_morphology": morphology_name,
                    "bbox": [x1, y1, x2, y2],
                    "score": round(score, 2)
                })

            if dental_morphology_types:
                detections = [det for det in detections if det['dental_morphology'] in dental_morphology_types]

            # Create output object
            viz_output = IntraoralImageToothTypeDetectionOutput(detections=detections)

            # Save visualization
            viz_path = None
            viz_path = self._save_visualization(
                image_path=image_path,
                output=viz_output
            )

            # Prepare output and metadata
            output = {
                "detection_image_path": viz_path,
                "detections": detections,
            }

            metadata = {
                "image_path": image_path,
                "detection_image_path": viz_path,
                "original_size": orig_size,
                "model_size": tuple(image_tensor.shape[-2:]),
                "confidence_threshold": self.confidence,
                "detected_dental_morphologies": list(set(d["dental_morphology"] for d in detections)),
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
        dental_morphology_types: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_path, dental_morphology_types)

    @staticmethod
    def _convert_bbox_to_xyxy(box_orig):
        """Convert bbox from cx, cy, w, h to xmin, ymin, xmax, ymax."""
        cx, cy, w, h = box_orig
        x1 = round(float(cx - w / 2))
        y1 = round(float(cy - h / 2))
        x2 = round(float(cx + w / 2))
        y2 = round(float(cy + h / 2))
        return [x1, y1, x2, y2]

    def _save_visualization(self, image_path: str, output: IntraoralImageToothTypeDetectionOutput) -> str:
        """
        Visualize the dental morphology detection results and save the visualization as an image.

        Args:
            image_path (str): Path to the input intraoral image.
            output (IntraoralImageToothTypeDetectionOutput): Detection output containing dental morphology type, bounding boxes, and scores.
        """
        # Load the original image
        orig_image = Image.open(image_path).convert("RGB")
        fig, ax = plt.subplots(1, figsize=(12, 10))
        ax.imshow(orig_image)
        ax.axis('off')
        ax.set_title(f"Dental Morphology Detection: {os.path.basename(image_path)}")

        unique_types = list(set(d['dental_morphology'] for d in output.detections))
        color_map = {t: plt.cm.rainbow(i / max(len(unique_types), 1)) for i, t in enumerate(unique_types)}

        for detection in output.detections:
            bbox = detection['bbox']
            morphology_type = detection['dental_morphology']
            score = detection['score']

            x_min, y_min, x_max, y_max = bbox
            color = color_map[morphology_type]

            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

            label = f"{morphology_type} ({score:.2f})"
            ax.text(
                x_min, y_min - 5, label,
                fontsize=9, color='white',
                bbox=dict(facecolor=color, alpha=0.5)
            )

        # Save the visualization
        save_path = self.temp_dir / f"detection_ToothType_{uuid.uuid4().hex[:8]}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to: {save_path}")
        return str(save_path)