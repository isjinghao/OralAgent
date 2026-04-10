from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import skimage.io
import skimage.measure
import skimage.transform
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
import inspect
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import uuid
import json


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from ..model_DINO.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

class PanoramicXRayPeriapicalLesionSubClassDetectionInput(BaseModel):
    """Input schema for the Panoramic X-ray Periapical Lesion Subclass Detection Tool."""

    image_path: str = Field(..., description="Path to the Panoramic X-ray image file to be processed")
    confidence: Optional[float] = Field(0.5, description="Confidence threshold for detection")
    periapical_lesion_names: Optional[List[str]] = Field(
        None,
        description="A list of periapical lesion subclass names to detect. If set to None, all available subclasses will be detected. "
        "The available subclasses include: Granuloma, Cyst, and Abscess. "
    )

class PanoramicXRayPeriapicalLesionSubClassDetectionOutput(BaseModel):
    """Output schema for DINO Periapical Lesion Subclass Detection Tool."""
    detections: List[Dict[str, Any]] = Field(..., description="List of detected periapical lesion subclasses and bounding boxes")


class PanoramicXRayPeriapicalLesionSubClassDetectionTool(BaseTool):
    """Tool for performing detailed periapical lesion subclass detection analysis of panoramic X-ray images."""

    name: str = "panoramic_xray_periapical_lesion_subclass_detection"
    description: str = (
        "Detects periapical lesion subclasses in panoramic X-ray images. "
        "It identifies specific lesion subclasses, such as Granuloma, Cyst, and Abscess. "
        "Returns detection visualization and a list of detected lesion subclasses with their bounding boxes and confidence scores. "
        "Ensure the input image is of high quality for optimal performance."
    )

    args_schema: Type[BaseModel] = PanoramicXRayPeriapicalLesionSubClassDetectionInput

    config_path: str = ""
    checkpoint_path: str = ""
    coco_names_path: str = ""
    device: Optional[str] = "cuda"
    transform: Any = None
    temp_dir: Path = Path("temp")
    model: Any = None
    postprocessors: Any = None
    id2name: Any = None

    def __init__(self, config_path: str, checkpoint_path: str, coco_names_path: str, device: Optional[str] = "cuda", temp_dir: Optional[Path] = Path("temp")):
        """Initialize the DINO Periapical Lesion Subclass Detection Tool."""
        super().__init__()
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.coco_names_path = coco_names_path
        self.device = torch.device(device) if device else "cuda"
        self.model, self.postprocessors = self._load_model()
        self.id2name = self._load_category_names()
        self.temp_dir = temp_dir if isinstance(temp_dir, Path) else Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def _load_model(self):
        """Load the DINO model."""
        args = SLConfig.fromfile(self.config_path)
        args.device = self.device
        model, _, postprocessors = build_model_main(args)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
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
            orig_size = image.size
            image_transformed, _ = transform(image, None)
            return image_transformed, orig_size

    def _run_inference(self, image_tensor):
        """Run inference on the input image tensor."""
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            outputs = self.model(image_tensor)
            processed_outputs = self.postprocessors['bbox'](outputs, torch.tensor([[1.0, 1.0]]).to(self.device))[0]
        return processed_outputs

    def _run(self, 
            image_path: str,
            confidence: Optional[float] = 0.5,
            periapical_lesion_names: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None,
            ) -> PanoramicXRayPeriapicalLesionSubClassDetectionOutput:
        try:
            """Run the DINO Periapical Lesion Subclass Detection Tool."""
            # Preprocess the image
            image_tensor, orig_size = self._preprocess_image(image_path)
            orig_w, orig_h = orig_size

            # Run inference
            outputs = self._run_inference(image_tensor)

            # Process detections
            detections = []
            for i in range(len(outputs['boxes'])):
                box_cxcywh = box_ops.box_xyxy_to_cxcywh(outputs['boxes'][i]).cpu().numpy()
                cx, cy, w, h = box_cxcywh
                box_orig = [
                    cx * orig_w,
                    cy * orig_h,
                    w * orig_w,
                    h * orig_h
                ]
                label = outputs['labels'][i].item()
                score = outputs['scores'][i].item()

                if score < confidence:
                    continue

                lesion_subclass_name = self.id2name.get(label, f"Unknown ({label})")
                detections.append({
                    "lesion_subclass_name": lesion_subclass_name,
                    "bbox": self._convert_bbox_to_xyxy(box_orig),
                    "score": round(score, 2)
                })

            if periapical_lesion_names:
                detections = [det for det in detections if det['lesion_subclass_name'] in periapical_lesion_names]

            # Create output object
            output = PanoramicXRayPeriapicalLesionSubClassDetectionOutput(detections=detections)

            # Save visualization
            viz_path = self._save_visualization(
                    image_path=image_path,
                    output=output
                    )

            # convert boxes for detected teeth
            results = {}
            for detection in detections:
                bbox = detection['bbox']
                score = detection['score']
                lesion_subclass_name = detection['lesion_subclass_name']
                results[lesion_subclass_name] = {'box': bbox, 'score': score}

            # Prepare output and metadata
            output = {
                "detection_image_path": viz_path,
                "detections": detections,
                # "metrics": {tooth_id: metrics.dict() for tooth_id, metrics in results.items()},
            }

            metadata = {
                "image_path": image_path,
                "detection_image_path": viz_path,
                "original_size": orig_size,
                "model_size": tuple(image_tensor.shape[-2:]),
                "confidence_threshold": confidence,
                "processed_teeth": list(results.keys()),
                "analysis_status": "completed",
            }
            return output, metadata

        except Exception as e:
            # Handle errors and prepare error output and metadata
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
        confidence: float,
        tooth_ids: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_path, confidence, tooth_ids)

    @staticmethod
    def _convert_bbox_to_xyxy(box_orig):
        """Convert bbox from cx, cy, w, h to xmin, ymin, xmax, ymax."""
        cx, cy, w, h = box_orig
        x1 = round(float(cx - w / 2))
        y1 = round(float(cy - h / 2))
        x2 = round(float(cx + w / 2))
        y2 = round(float(cy + h / 2))
        return [x1, y1, x2, y2]

    def _save_visualization(self, image_path: str, output: PanoramicXRayPeriapicalLesionSubClassDetectionOutput) -> str:
        """
        Visualize the detection results and save the visualization as an image.

        Args:
            image_path (str): Path to the input image.
            output (PanoramicXRayPeriapicalLesionSubClassDetectionOutput): Detection output containing lesion subclass name, bounding boxes, and scores.
        """
        # Load the original image
        orig_image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = orig_image.size

        # Create a figure and axis for visualization
        fig, ax = plt.subplots(1, figsize=(12, 10))
        ax.imshow(orig_image)
        ax.axis('off')
        ax.set_title(f"Periapical Lesion Subclass Detection Results: {os.path.basename(image_path)}")

        # Use different colors for different teeth
        colors = plt.cm.rainbow(np.linspace(0, 1, len(output.detections)))

        # Draw each detection
        for i, detection in enumerate(output.detections):
            bbox = detection['bbox']  # [xmin, ymin, xmax, ymax]
            lesion_subclass_name = detection['lesion_subclass_name']
            score = detection['score']

            # Extract bounding box coordinates
            x_min, y_min, x_max, y_max = bbox

            # Draw the bounding box
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor=colors[i % len(colors)], facecolor='none'
            )
            ax.add_patch(rect)

            # Add a label with the tooth ID and confidence score
            label = f"{lesion_subclass_name} ({score:.2f})"
            ax.text(
                x_min, y_min - 5, label,
                fontsize=9, color='white',
                bbox=dict(facecolor=colors[i % len(colors)], alpha=0.5)
            )

        # Save the visualization
        save_path = self.temp_dir / f"detection_{uuid.uuid4().hex[:8]}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to: {save_path}")
        return str(save_path)
