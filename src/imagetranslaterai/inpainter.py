import os
import requests
import logging
import cv2
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)

class Inpainter:
    def __init__(self):
        self.api_key = os.getenv("STABILITY_API_KEY")
        if not self.api_key:
            logger.warning("STABILITY_API_KEY not found. Inpainting will likely fail if using API.")
        self.api_host = "https://api.stability.ai"

    def create_mask(self, image_path: str, boxes: List[List[List[float]]], padding: int = 2) -> str:
        """
        Creates a refined mask (Minimal Masking).
        padding: Reduced to 2 (from 5) for Surgical Precision.
        Applies GaussianBlur for Soft Edges.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
            
        h, w = img.shape[:2]
        # 검은 배경
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for box in boxes:
            pts = np.array(box, dtype=np.int32)
            # 1. Fill the polygon (White)
            cv2.fillPoly(mask, [pts], 255)
            
            # 2. Draw thicker contours
            if padding > 0:
                cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=padding*2)

        # 3. [New] Gaussian Blur for Soft Edges
        # 마스크 경계가 칼같이 끊기면 합성 티가 납니다. 부드럽게 처리합니다.
        mask = cv2.GaussianBlur(mask, (7, 7), 0) # Kernel size 7x7

        # Output path
        mask_path = os.path.splitext(image_path)[0] + "_mask.png"
        cv2.imwrite(mask_path, mask)
        logger.info(f"Refined mask (Soft Edge) created at: {mask_path}")
        return mask_path

    def inpaint(self, image_path: str, mask_path: str, output_path: str) -> str:
        """
        Uses Stability AI to inpaint.
        PROMPT ENGINEERING UPDATED: Focus on 'texture preserving' rather than 'remove text'.
        """
        if not self.api_key:
            logger.warning("No Stability API key. Falling back to OpenCV inpainting.")
            return self.inpaint_cv2(image_path, mask_path, output_path)

        logger.info(f"Sending inpainting request for {image_path}...")
        
        try:
            with open(image_path, "rb") as f_img, open(mask_path, "rb") as f_mask:
                # --- [핵심 수정] 프롬프트 강화 (Texture Focus) ---
                prompt = (
                    "high details, preserving wall texture, sharp focus, "
                    "fluid background texture, match surrounding gradient, "
                    "no text, no watermark"
                )
                
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
                        "search_prompt": "text", 
                        "output_format": "png",  
                        "mask_blur": 5, # [New] AI-side blur for blending
                    },
                )

            if response.status_code == 200:
                with open(output_path, 'wb') as file:
                    file.write(response.content)
                logger.info(f"Stability Inpainted image saved to {output_path}")
                return output_path
            else:
                logger.error(f"Stability AI Error: {response.json()}")
                # API 실패 시 OpenCV로 폴백
                return self.inpaint_simple_fill(image_path, mask_path, output_path)
                
        except Exception as e:
            logger.error(f"Stability Inpainting failed: {e}. Trying Simple Fill fallback.")
            return self.inpaint_simple_fill(image_path, mask_path, output_path)

    def inpaint_simple_fill(self, image_path: str, mask_path: str, output_path: str) -> str:
        """
        Fallback: OpenCV Telea.
        Uses a smaller radius to preserve details.
        """
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)
        
        if img is None or mask is None:
            raise ValueError("Could not load image or mask")
            
        # Ensure mask covers edges
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Telea works better for small text removal
        radius = 3 
        inpainted_img = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)

        cv2.imwrite(output_path, inpainted_img)
        logger.info(f"OpenCV (Fallback) image saved to {output_path}")
        return output_path

    def inpaint_cv2(self, image_path, mask_path, output_path):
        return self.inpaint_simple_fill(image_path, mask_path, output_path)