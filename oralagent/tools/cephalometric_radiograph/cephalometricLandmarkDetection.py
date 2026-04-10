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


############################### for CeLDA Detection Model ##################################
import json
import os
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import tabulate
import albumentations as A
from ..model_CeLDA.Unet import UNet2D
import argparse
import cv2
from PIL import Image


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class CephalometricXRayLandmarkDetectionInput(BaseModel):
    """Input schema for the Cephalometric X-ray Landmark Detection Tool."""

    image_path: str = Field(..., description="Path to the Panoramic X-ray image file to be processed")
    landmark_names: Optional[List[str]] = Field(
        None,
        description="A list of specific landmark names to detect."
            "If set to None, the tool will detect all available landmarks. Users can specify "
            "one or more landmark names to focus the detection on specific anatomical points. "
    )

class CephalometricXRayLandmarkDetectionOutput(BaseModel):
    """Output schema for the Cephalometric X-ray Landmark Detection Tool."""

    detections: List[Dict[str, Any]] = Field(..., description="List of detected cephalometric landmarks, including their names and coordinates. ")


class CephalometricXRayLandmarkDetectionTool(BaseTool):
    """Tool for performing detailed cephalometric landmark detection analysis of cephalometric X-ray images."""


    name: str = "cephalometric_xray_landmark_detection"
    description: str = (
        "Detects anatomical landmarks in cephalometric X-ray images. "
        "It identifies specific cephalometric landmarks, such as Nasion, Sella, and other key points used in orthodontic analysis. "
        "The tool provides a visualization of the detected landmarks overlaid on the input image, along with their coordinates. "
        "Ensure the input cephalometric X-ray image is of high resolution and quality for accurate landmark detection."
    )

    args_schema: Type[BaseModel] = CephalometricXRayLandmarkDetectionInput

    checkpoint_path: str = ""
    prototype_path: str = ""

    coco_names_path: str = ""

    device: Optional[str] = "cuda"
    transform: Any = None
    temp_dir: Path = Path("temp")

    model: Any = None
    image_size: Any = None
    id2name: Any = None

    def __init__(self, checkpoint_path: str, prototype_path: str, coco_names_path: str, image_size: Tuple[int, int] = (512, 512), device: Optional[str] = "cuda", temp_dir: Optional[Path] = Path("temp")):
        """Initialize the Cephalometric Landmark Detection Tool."""
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.prototype_path = prototype_path
        self.coco_names_path = coco_names_path
        self.image_size = image_size
        self.device = torch.device(device) if device else "cuda"
        self.model = self._load_model(self.checkpoint_path, self.prototype_path)
        
        self.id2name = self._load_category_names()

        self.temp_dir = temp_dir if isinstance(temp_dir, Path) else Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def _load_model(self, checkpoint_path, prototype_path):
        net = UNet2D(in_channels=3, out_channels=1)
        state_dict = torch.load(checkpoint_path, weights_only=True)
        net.load_state_dict(state_dict)
        prototype = torch.load(prototype_path, weights_only=True)
        net.set_prtotype(prototype)
        net.eval()
        return net

    def _read_image(self, img_path, aug):
        image = cv2.imread(img_path)
        raw_shape = image.shape[:-1]
        augmentation = aug(image=image)
        image = augmentation['image']
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        return image, raw_shape

    # def _mapping_back(self, pred, raw_shape, image_size):
    #     # resize with max size 2048
    #     index = np.argmax(raw_shape)
    #     scale = 2048 / raw_shape[index]
    #     aug = A.Compose([
    #                         A.Resize(int(raw_shape[0]*scale), int(raw_shape[1]*scale)),
    #                     ], keypoint_params=A.KeypointParams(format='yx'))
    #     pred = aug(image=np.zeros((image_size,image_size,3)), keypoints=pred)['keypoints']

    #     return np.array(pred)
    
    def _mapping_back(self, pred, raw_shape, image_size):
        # 计算宽度和高度的缩放比例
        scale_x = raw_shape[1] / image_size
        scale_y = raw_shape[0] / image_size

        # 将关键点坐标映射回原始图像尺寸
        pred = np.array([[x * scale_x, y * scale_y] for x, y in pred])

        return pred


    def _load_category_names(self):
        """Load category names from COCO format."""
        with open(self.coco_names_path, 'r') as f:
            categories = json.load(f)
        return {int(cat_id): cat_name for cat_id, cat_name in categories.items()}

    def _preprocess_image(self, image_path: str):
            """Preprocess the input image."""
            transform = A.Compose(
                [A.Resize(self.image_size[0], self.image_size[1]), A.Normalize()],
            )
            image, raw_shape = self._read_image(image_path, transform)
            return image, raw_shape

    def _run_inference(self, image_tensor, raw_shape):
        """Run inference on the input image tensor."""
        self.model.to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor.to(self.device))

        features = [F.interpolate(feature_output, size=image_tensor.shape[-2:], mode='bilinear', align_corners=True) for feature_output in features]
        features = torch.cat(features, dim=1).squeeze().cpu()
        similarity = torch.einsum('bi,ijk->bjk', self.model.prototype.cpu().float(), features.float()).numpy()
        pred_raw = []
        for ii in range(len(similarity)):
            threshold = np.percentile(similarity[ii], 99.95)
            similarity[ii][similarity[ii] < threshold] = 0
            tmp = np.mean(np.argwhere(similarity[ii] > 0), axis=0)
            pred_raw.append([tmp[1], tmp[0]])
        pred_points = self._mapping_back(pred_raw, raw_shape, self.image_size[0])

        return pred_points

    def _run(self, 
            image_path: str,
            landmark_names: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None,
            ) -> CephalometricXRayLandmarkDetectionOutput:
        try:
            """Run the Cephalometric Landmark Detection Tool."""

            # Preprocess the image
            image_tensor, orig_size = self._preprocess_image(image_path)
            # Run inference
            pred_all_points = self._run_inference(image_tensor, orig_size)

            # Process detections
            landmarks = []
            for i in range(pred_all_points.shape[0]):
                x, y = pred_all_points[i]
                landmark_name = self.id2name.get(i, f"Unknown ({i})")
                landmarks.append({
                    "landmark_name": landmark_name,
                    "coordinates": (round(float(x)), round(float(y)))
                })

            if landmark_names:
                landmarks = [lm for lm in landmarks if lm['name'] in landmark_names]

            # # Create output object
            output = CephalometricXRayLandmarkDetectionOutput(detections=landmarks)

            # Save visualization
            viz_path = None
            # viz_path = self._save_visualization(
            #         image_path=image_path,
            #         pred_all_points=pred_all_points,
            #         )

            # Prepare output and metadata
            output = {
                "detection_image_path": viz_path,
                "landmarks": landmarks,
            }

            metadata = {
                "image_path": image_path,
                "detection_image_path": viz_path,
                "original_size": orig_size,
                "model_size": tuple(image_tensor.shape[-2:]),
                "processed_landmarks": landmark_names,
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
        landmark_names: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_path, landmark_names)

    def _save_visualization(self, image_path: str, pred_all_points: Any) -> str:
        """
        Save a visualization of the predictions for cephalometric landmark detection.
        """
        # Load the original image
        orig_image = Image.open(image_path).convert("RGB")

        # Create a figure and axis for visualization
        fig, ax = plt.subplots(1, figsize=(12, 10))
        ax.imshow(orig_image)
        ax.axis('off')
        ax.set_title(f"Cephalometric Landmark Detection Results: {os.path.basename(image_path)}")
        for index, pred in enumerate(pred_all_points):
            plt.scatter(pred[0], pred[1], c='green', s=75, label='Predicted' if index == 0 else "")

        plt.axis('off')
        # Save the visualization
        save_path = self.temp_dir / f"landmarks_{uuid.uuid4().hex[:8]}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0) 
        plt.close()
        
        print(f"Visualization saved to: {save_path}")
        return str(save_path)