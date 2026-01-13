import os
import logging
import json
from imagetranslaterai.ocr_engine import OCREngine
from imagetranslaterai.translator import Translator
from imagetranslaterai.inpainter import Inpainter
from imagetranslaterai.renderer import TextRenderer
from imagetranslaterai.utils import download_image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Setup
    assets_dir = os.path.join(os.getcwd(), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # Use a simpler image for full pipeline test if possible, or keep the poster
    image_url = "https://raw.githubusercontent.com/paddlepaddle/PaddleOCR/release/2.6/doc/imgs/11.jpg"
    image_path = os.path.join(assets_dir, "pipeline_test_original.jpg")
    
    if not os.path.exists(image_path):
        download_image(image_url, image_path)

    try:
        # 2. OCR
        logger.info(">>> Step 1: OCR Detection")
        ocr = OCREngine()
        ocr_results = ocr.detect_text(image_path)
        
        if not ocr_results:
            logger.warning("No text detected. Aborting pipeline.")
            return

        # Extract only boxes for inpainting
        all_boxes = [item['box'] for item in ocr_results]

        # 3. Inpainting (Background Restoration)
        logger.info(">>> Step 2: Background Inpainting")
        inpainter = Inpainter()
        
        # Create mask
        mask_path = inpainter.create_mask(image_path, all_boxes, padding=3)
        
        # Perform inpainting
        inpainted_path = os.path.join(assets_dir, "pipeline_restored_bg.webp")
        # Inpainter requires API key, handle gracefully if missing or error
        try:
            # Use simple fill directly as requested by user
            inpainter.inpaint_simple_fill(image_path, mask_path, inpainted_path)
            logger.info(f"Background restored (Simple Fill): {inpainted_path}")
        except Exception as e:
            logger.error(f"Skipping inpainting: {e}")
            inpainted_path = image_path # Fallback

        # 4. Translation & Analysis
        logger.info(">>> Step 3: Translation & Style Analysis")
        translator = Translator()
        analysis_data = [] # Initialize
        try:
            analysis_data = translator.translate_and_analyze(ocr_results, image_path)
            
            # Save analysis to JSON for review
            analysis_path = os.path.join(assets_dir, "translation_analysis.json")
            with open(analysis_path, "w", encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=4, ensure_ascii=False)
            
            logger.info("Translation analysis success.")
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")

        # 5. Final Rendering
        if analysis_data:
            logger.info(">>> Step 4: Final Rendering")
            renderer = TextRenderer()
            
            # Decide which background to use: restored one if available, else original
            bg_to_use = inpainted_path if (os.path.exists(inpainted_path) and inpainted_path != image_path) else image_path
            final_output_path = os.path.join(assets_dir, "final_output.jpg")
            
            try:
                renderer.render_text(bg_to_use, analysis_data, final_output_path)
            except Exception as e:
                logger.error(f"Rendering failed: {e}")
    
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
