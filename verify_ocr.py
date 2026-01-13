import os
import logging
from imagetranslaterai.ocr_engine import OCREngine
from imagetranslaterai.utils import download_image

# Setup simple logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Setup paths
    assets_dir = os.path.join(os.getcwd(), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # Use a sample poster image (e.g., a movie poster or product ad with text)
    # Using a placeholder URL for a tech conference poster or similar with clear text
    image_url = "https://raw.githubusercontent.com/paddlepaddle/PaddleOCR/release/2.6/doc/imgs/11.jpg" # Standard PaddleOCR sample
    save_path = os.path.join(assets_dir, "sample_poster.jpg")
    
    # 2. Download Image
    if not os.path.exists(save_path):
        logger.info("Downloading sample image...")
        download_image(image_url, save_path)
    else:
        logger.info("Sample image already exists.")

    # 3. Initialize OCR Engine
    logger.info("Initializing OCR Engine...")
    ocr_engine = OCREngine(lang='korean') # Defaulting to Korean mode (supports English too)

    # 4. Run Detection
    logger.info("Running text detection...")
    try:
        results = ocr_engine.detect_text(save_path)
        
        # 5. Print Results
        print("\n" + "="*50)
        print(f"OCR Results for: {os.path.basename(save_path)}")
        print("="*50)
        
        if not results:
            print("No text detected.")
        
        for i, item in enumerate(results):
            text = item['text']
            score = item['score']
            box = item['box']
            print(f"[{i+1}] Text: {text} (Conf: {score:.4f})")
            # print(f"    Box: {box}") # Uncomment if bounding box details are needed
            
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")

if __name__ == "__main__":
    main()
