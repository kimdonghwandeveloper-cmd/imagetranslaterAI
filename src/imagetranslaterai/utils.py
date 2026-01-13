import requests
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def download_image(url: str, save_path: str) -> Optional[str]:
    """
    Download an image from a URL and save it to the specified path.
    
    Args:
        url (str): The URL of the image.
        save_path (str): The local path to save the image.
        
    Returns:
        Optional[str]: The absolute path to the saved file if successful, else None.
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
