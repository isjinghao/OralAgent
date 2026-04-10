from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid
import tempfile

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


############################### for MaskDINO Segmentation Model ##################################
import os
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from ..model_MaskDINO import add_maskdino_config
from ..model_MaskDINO.predictor import VisualizationDemo
from typing import Any, Dict, List, Optional, Type
import time
import json
import random
from detectron2.data import Metadata


class PeriapicalXRayDiseaseSegmentationInput(BaseModel):
    """Input schema for the Periapical X-ray Disease Segmentation Tool."""
    image_path: str = Field(..., description="Path to the Periapical X-ray image file to be processed")
    diseases: Optional[List[str]] = Field(
        None,
        description="A list of disease names to detect in periapical X-ray images. If set to None, the tool will detect all available diseases. "
        "The available diseases include: "
        "Crown, Caries, Root Canal Treatment, Restoration, Normal. "
        "This list allows users to specify targeted diseases for detection or perform a comprehensive analysis of all supported categories."
    )

class PeriapicalXRayDiseaseSegmentationOutput(BaseModel):
    """Output schema for MaskDINO Disease Segmentation Tool."""
    segments: List[Dict[str, Any]] = Field(..., description="List of segmented regions with their properties")


class PeriapicalXRayDiseaseSegmentationTool(BaseTool):
    """Tool for performing disease segmentation analysis of periapical X-ray images using MaskDINO."""

    name: str = "periapical_xray_disease_segmentation"
    description: str = (
        "Detects and segments regions of diseases and anatomical structures in periapical X-ray images."
        "This tool can identify the following categories: "
        "Crown, Caries, Root Canal Treatment, Restoration, Normal. "
        "Returns segmentation visualization for each detected category."
    )

    args_schema: Type[BaseModel] = PeriapicalXRayDiseaseSegmentationInput
    temp_dir: Path = Path("temp")
    cfg: Any = None
    demo: Any = None
    device: Optional[str] = "cuda"
    coco_names_path: str = ""
    id2name: Any = None
    category_metadata: Any = None

    def __init__(self, config_path: str, checkpoint_path: str, coco_names_path: str, confidence_threshold: float = 0.5, device: Optional[str] = "cuda", temp_dir: Optional[Path] = Path("temp")):
        """Initialize the MaskDINO Disease Segmentation Tool."""
        super().__init__()
        self.cfg = self._setup_cfg(config_path, checkpoint_path, confidence_threshold)
        self.coco_names_path = coco_names_path
        self.id2name = self._load_category_names()
        self.category_metadata = self._load_category_metadata('panoramic_X-ray_11diseases')
        self.demo = VisualizationDemo(self.cfg, self.category_metadata)
        self.device = device
        self.temp_dir = temp_dir if isinstance(temp_dir, Path) else Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
    def _setup_cfg(self, config_path: str, checkpoint_path: str, confidence_threshold: float):
        """Set up the configuration for MaskDINO."""
        cfg = get_cfg()
        cfg.confidence_threshold = confidence_threshold
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = checkpoint_path

        cfg.freeze()
        return cfg


    def _run(
        self,
        image_path: str,
        diseases: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Run segmentation analysis for specified diseases (currently supports 6 diseases and anatomical structures in periapical X-ray images)."""
        try:
            # Validate and get disease indices
            supportive_vategories = list(self.id2name.values())

            if diseases:
                diseases = [d.strip() for d in diseases]
                invalid_diseases = [d for d in diseases if d not in supportive_vategories]
                if invalid_diseases:
                    raise ValueError(f"Invalid diseases specified: {invalid_diseases}")
            else:
                diseases = supportive_vategories

            # Read the image
            img = read_image(image_path, format="BGR")

            # Run inference
            start_time = time.time()
            predictions, visualized_output = self.demo.run_on_image(img, self.cfg.confidence_threshold)
            inference_time = time.time() - start_time

            # Process predictions
            results = []
            if "instances" in predictions:
                instances = predictions["instances"].to("cpu")
                masks = instances.pred_masks.numpy()
                scores = instances.scores.numpy()
                labels = instances.pred_classes.numpy()

                for mask, score, label in zip(masks, scores, labels):
                    category_name = self.id2name[label]
                    if category_name not in diseases:
                        continue
                    
                    # 获取外接四边形
                    y_indices, x_indices = np.where(mask)  # 获取 mask 中非零像素的坐标
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        x_min, x_max = x_indices.min(), x_indices.max()
                        y_min, y_max = y_indices.min(), y_indices.max()
                        bbox = [x_min, y_min, x_max, y_max]  # [xmin, ymin, xmax, ymax]
                    else:
                        bbox = None  # 如果 mask 为空，则返回默认值

                    if bbox is not None:
                        results.append({
                            "label": category_name,
                            "bbox": bbox,
                            "score": round(float(score), 2)
                        })

            print(f"results: {results}")

            # Save visualization
            viz_path = None
            # viz_path = os.path.join(self.temp_dir, f"visualized_{uuid.uuid4().hex[:8]}.png")
            # os.makedirs(os.path.dirname(viz_path), exist_ok=True)
            # visualized_output.save(viz_path)
            # print(f"Visualization saved to: {viz_path}")

            # Prepare output and metadata
            output = {
                "segmentation_image_path": viz_path,
                "results": results,
            }

            metadata = {
                "image_path": image_path,
                "segmentation_image_path": viz_path,
                "inference_time": inference_time,
                "original_size": img.shape[:2],
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
        diseases: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_path, diseases)

    def _load_category_names(self):
        """Load category names from COCO format."""
        with open(self.coco_names_path, 'r') as f:
            categories = json.load(f)
        return {int(cat_id): cat_name for cat_id, cat_name in categories.items()}

    def _load_category_metadata(self, name):
        # 随机生成颜色
        random.seed(42)  # 固定随机种子，确保每次生成的颜色一致
        thing_classes = list(self.id2name.values())
        thing_colors = [
            [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            for _ in thing_classes
        ]

        # 创建 Metadata 对象
        metadata = Metadata(name=name)
        metadata.thing_classes = thing_classes
        metadata.thing_colors = thing_colors

        return metadata