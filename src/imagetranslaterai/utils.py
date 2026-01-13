from PIL import Image
import logging
import os
import requests
from typing import Optional

logger = logging.getLogger(__name__)

class ImageUtils:
    @staticmethod
    def crop_image(image_path: str, x: int, y: int, w: int, h: int) -> str:
        """
        Crops the image to the specified ROI and saves it as a temporary file.
        """
        try:
            with Image.open(image_path) as img:
                cropped = img.crop((x, y, x + w, y + h))
                
                dir_name = os.path.dirname(image_path)
                file_name = os.path.splitext(os.path.basename(image_path))[0]
                crop_path = os.path.join(dir_name, f"{file_name}_crop_{x}_{y}.jpg")
                
                cropped.convert("RGB").save(crop_path)
                logger.info(f"ROI cropped to: {crop_path}")
                return crop_path
        except Exception as e:
            logger.error(f"Failed to crop image: {e}")
            raise

    @staticmethod
    def merge_image(original_path: str, processed_crop_path: str, x: int, y: int, output_path: str) -> str:
        """
        Merges the processed crop back into the original image at (x, y).
        """
        try:
            with Image.open(original_path) as orig:
                with Image.open(processed_crop_path) as crop:
                    orig_rgba = orig.convert("RGBA")
                    crop_rgba = crop.convert("RGBA")
                    
                    orig_rgba.paste(crop_rgba, (x, y), crop_rgba)
                    
                    final_img = orig_rgba.convert("RGB")
                    final_img.save(output_path)
                    logger.info(f"Merged ROI into final image: {output_path}")
                    return output_path
        except Exception as e:
            logger.error(f"Failed to merge image: {e}")
            raise

def download_image(url: str, save_path: str) -> Optional[str]:
    """
    Download an image from a URL and save it to the specified path.
    ... (Existing docstring) ...
    """
    try:
        logger.info(f"Downloading image from {url}")
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Image saved to {save_path}")
        return os.path.abspath(save_path)
    except Exception as e:
        logger.error(f"Failed to download image: {e}")
        return None
