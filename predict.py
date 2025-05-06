# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import cv2
import sys
import time
import torch
import subprocess
import numpy as np
from PIL import Image
from typing import List, Optional
from sam2.sam2_image_predictor import SAM2ImagePredictor
# Add /tmp/sa2 to sys path
sys.path.extend("/sa2")
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

WEIGHTS_CACHE = "checkpoints"
MODEL_NAME = "sam2_hiera_large.pt"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["wget", "-O", dest, url], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        os.chdir("/sa2")
        # Get path to model
        model_cfg = "sam2_hiera_l.yaml"
        model_path = WEIGHTS_CACHE + "/" +MODEL_NAME
        # Download weights
        if not os.path.exists(model_path):
            download_weights(WEIGHTS_URL, model_path)
        # Setup SAM2
        self.sam2 = build_sam2(config_file=model_cfg, ckpt_path=model_path, device='cuda', apply_postprocessing=False)
        # turn on tfloat32 for Ampere GPUs
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Initialize SAM2ImagePredictor
        self.predictor = SAM2ImagePredictor(self.sam2)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        point_coords: Optional[List[List[float]]] = Input(description="List of [x, y] point coordinates", default=None),
        point_labels: Optional[List[int]] = Input(description="List of point labels (1 for foreground, 0 for background)", default=None),
        box: Optional[List[float]] = Input(description="Box coordinates [x1, y1, x2, y2]", default=None),
        mask_input: Optional[Path] = Input(description="Path to mask input image (256x256)", default=None),
        multimask_output: bool = Input(description="Whether to output multiple masks", default=True),
        normalize_coords: bool = Input(description="Whether to normalize coordinates to [0,1]", default=True),
    ) -> List[Path]:
        # Load image
        image_pil = Image.open(image).convert('RGB')
        # Set image
        self.predictor.set_image(image_pil)
        # Prepare prompts
        point_coords_np = np.array(point_coords, dtype=np.float32) if point_coords is not None else None
        point_labels_np = np.array(point_labels, dtype=np.int32) if point_labels is not None else None
        box_np = np.array(box, dtype=np.float32) if box is not None else None
        if mask_input is not None:
            mask_pil = Image.open(mask_input).convert('L')
            mask_pil = mask_pil.resize((256, 256), Image.BILINEAR)
            mask_np = np.array(mask_pil, dtype=np.float32)[None, :, :]
        else:
            mask_np = None
        # Call predict
        masks, _, _ = self.predictor.predict(
            point_coords=point_coords_np,
            point_labels=point_labels_np,
            box=box_np,
            mask_input=mask_np,
            multimask_output=multimask_output,
            return_logits=False,
            normalize_coords=normalize_coords,
        )
        # Save masks
        return_masks = []
        for i, mask in enumerate(masks):
            mask_img = (mask * 255).astype(np.uint8)
            mask_path = Path(f"/tmp/mask_{i}.png")
            Image.fromarray(mask_img).save(mask_path)
            return_masks.append(mask_path)
        return return_masks