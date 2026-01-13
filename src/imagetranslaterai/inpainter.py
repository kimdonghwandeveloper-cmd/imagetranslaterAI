import os
import requests
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class Inpainter:
    def __init__(self):
        self.api_key = os.getenv("STABILITY_API_KEY")
        if not self.api_key:
            logger.warning("STABILITY_API_KEY not found. Inpainting will likely fail if using API.")
        self.api_host = "https://api.stability.ai"

    def create_mask(self, image_path: str, boxes: List[List[List[float]]], padding: int = 0) -> str:
        """
        Creates a black and white mask image. 
        Padding default 0 to prevent merging of close boxes.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
            
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for box in boxes:
            pts = np.array(box, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            
        if padding > 0:
            kernel = np.ones((padding, padding), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        mask_path = image_path.replace(".jpg", "_mask.png").replace(".png", "_mask.png")
        cv2.imwrite(mask_path, mask)
        logger.info(f"Mask created at: {mask_path}")
        return mask_path

    # ... (inpaint method skipped, unchanging) ...

    def inpaint_simple_fill(self, image_path: str, mask_path: str, output_path: str) -> str:
        """
        Uses OpenCV inpainting.
        Micro-dilation: Kernel (3,3), Iterations=1.
        """
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)
        
        if img is None or mask is None:
            raise ValueError("Could not load image or mask")
            
        # 1. Micro Dilation (3x3, iter=1)
        kernel_dilate = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        # 2. Telea Inpainting
        radius = 3 # Reduced radius for finer control
        inpainted_img = cv2.inpaint(img, dilated_mask, radius, cv2.INPAINT_TELEA)

        cv2.imwrite(output_path, inpainted_img)
        logger.info(f"OpenCV (Telea) Inpainted image saved to {output_path}")
        return output_path

    def inpaint(self, image_path: str, mask_path: str, output_path: str) -> str:
        """
        Uses Stability AI to inpaint the masked area. Fits to background.
        """
        if not self.api_key:
            logger.warning("No Stability API key. Falling back to OpenCV inpainting.")
            return self.inpaint_cv2(image_path, mask_path, output_path)

        logger.info(f"Sending inpainting request for {image_path}...")
        
        try:
            with open(image_path, "rb") as f_img, open(mask_path, "rb") as f_mask:
                # Use a more descriptive prompt for text removal
                prompt = "clean background, seamless texture, high resolution, remove text, minimalist"
                
                response = requests.post(
                    f"{self.api_host}/v2beta/stable-image/edit/inpaint",
                    headers={
                        "authorization": f"Bearer {self.api_key}",
                        "accept": "image/*"
                    },
                    files={
                        "image": f_img,
                        "mask": f_mask,
                    },
                    data={
                        "prompt": prompt,
                        "search_prompt": "text", # Optional: guides what to search for in sematic inpainting
                        "output_format": "webp",
                    },
                )

            if response.status_code == 200:
                with open(output_path, 'wb') as file:
                    file.write(response.content)
                logger.info(f"Inpainted image saved to {output_path}")
                return output_path
            else:
                logger.error(f"Stability AI Error: {response.json()}")
                raise Exception("Stability AI API failed")
                
        except Exception as e:
            logger.error(f"Stability Inpainting failed: {e}. Trying Simple Fill fallback.")
            return self.inpaint_simple_fill(image_path, mask_path, output_path)

    def inpaint_simple_fill(self, image_path: str, mask_path: str, output_path: str) -> str:
        """
        Uses OpenCV inpainting methods which are better than simple color fill for gradients/textures.
        Also dilates mask slightly to ensure text edges are covered.
        """
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0) # Grayscale
        
        if img is None or mask is None:
            raise ValueError("Could not load image or mask")
            
        # 1. Micro Dilation (3x3, iter=1)
        kernel_dilate = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        # 2. Use Navier-Stokes or Telea based inpainting
        # Telea (INPAINT_TELEA) treats the point to be inpainted as a weighted average of known neighbors.
        # NS (INPAINT_NS) uses fluid dynamics equations. 
        # Telea is often sharper/better for text removal on simple backgrounds.
        radius = 5 # Inpainting radius
        inpainted_img = cv2.inpaint(img, dilated_mask, radius, cv2.INPAINT_TELEA)

        cv2.imwrite(output_path, inpainted_img)
        logger.info(f"OpenCV (Telea) Inpainted image saved to {output_path}")
        return output_path

    def inpaint_cv2(self, image_path: str, mask_path: str, output_path: str) -> str:
        """
        Simple local inpainting using OpenCV Telea algorithm.
        Good for simple solid backgrounds.
        """
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0) # Read as grayscale
        
        if img is None or mask is None:
            raise ValueError("Could not load image or mask for OpenCV inpainting")

        # Inpaint
        # Radius 3, flag INPAINT_TELEA or INPAINT_NS
        res = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        
        cv2.imwrite(output_path, res)
        logger.info(f"OpenCV Inpainted image saved to {output_path}")
        return output_path
