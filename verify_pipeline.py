import os
import logging
import json
import time
import cv2
import numpy as np
from imagetranslaterai.ocr_engine import OCREngine
from imagetranslaterai.translator import Translator
from imagetranslaterai.inpainter import Inpainter
from imagetranslaterai.renderer import TextRenderer
from imagetranslaterai.utils import ImageUtils # [ROI Add]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # 1. Setup
        assets_dir = os.path.join(os.getcwd(), "assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        # Use local image directly
        # Use local image directly
        # original_image_path = os.path.join(assets_dir, "sample_poster.jpg")
        original_image_path = os.path.join(assets_dir, "chinese_test.png")
        
        if not os.path.exists(original_image_path):
            logger.warning(f"Test image not found at {original_image_path}.")
            return

        # *** ROI SETTINGS (Example) ***
        # roi_coords = (50, 100, 800, 400) # (x, y, w, h) - Set to None to disable
        # *** ROI SETTINGS (Example) ***
        # roi_coords = (50, 100, 800, 400) # (x, y, w, h) - Set to None to disable
        # roi_coords = (50, 100, 800, 400) # (x, y, w, h) - Set to None to disable
        roi_coords = None # Full Image Processing
        
        # [ROI] Logic Switch
        if roi_coords:
            logger.info(f"ROI Processing Enabled: {roi_coords}")
            # Crop ROI from original
            processing_image_path = ImageUtils.crop_image(original_image_path, *roi_coords)
        else:
            logger.info("Full Image Processing Enabled")
            processing_image_path = original_image_path

        # 2. OCR
        logger.info(">>> Step 1: OCR Detection")
        # [Chinese Support] Change lang to 'ch'
        ocr = OCREngine(lang='ch') 
        ocr_results = ocr.detect_text(processing_image_path)
        
        if not ocr_results:
            logger.warning("No text detected. Aborting pipeline.")
            return

        # Visualize detected boxes
        try:
            debug_img = cv2.imread(processing_image_path)
            if debug_img is not None:
                for item in ocr_results:
                    # Draw box (Red, thickness 2)
                    box = np.array(item['box'], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(debug_img, [box], True, (0, 0, 255), 2)
                
                debug_box_path = os.path.join(assets_dir, "debug_ocr_boxes.jpg")
                cv2.imwrite(debug_box_path, debug_img)
                logger.info(f"Saved OCR visualization to {debug_box_path}")
        except Exception as e:
            logger.warning(f"Could not save debug image: {e}")

        # Extract only boxes for inpainting
        all_boxes = [item['box'] for item in ocr_results]

        # 3. Inpainting (Background Restoration)
        logger.info(">>> Step 2: Background Inpainting")
        inpainter = Inpainter()
        
        # Create mask
        # padding=5 (Ultra-Tight OCR에 맞춰 마스크영역 약간 확대)
        # Create mask
        # padding=5 (Ultra-Tight OCR에 맞춰 마스크영역 약간 확대)
        mask_path = inpainter.create_mask(processing_image_path, all_boxes, padding=2) # [Refactor] Padding 2
        
        # Perform inpainting
        inpainted_path = os.path.join(assets_dir, "pipeline_restored_bg.webp")
        # Inpainter requires API key, handle gracefully if missing or error
        try:
            # Use simple fill directly as requested by user or verify fallback
            inpainter.inpaint_simple_fill(processing_image_path, mask_path, inpainted_path)
            logger.info(f"Background restored (Simple Fill): {inpainted_path}")
        except Exception as e:
            logger.error(f"Skipping inpainting: {e}")
            inpainted_path = processing_image_path # Fallback

        # 4. Translation & Analysis (Multilingual Support)
        logger.info(">>> Step 3: Translation & Style Analysis")
        translator = Translator()
        
        # *** TARGET LANGUAGE SETTING ***
        target_lang = "Korean" # Changed to Korean for Chinese source
        logger.info(f"Target Language: {target_lang}")
        
        analysis_data = []
        try:
            analysis_data = translator.translate_and_analyze(ocr_results, processing_image_path, target_language=target_lang)
            
            # Save analysis to JSON for review
            analysis_path = os.path.join(assets_dir, "translation_analysis.json")
            with open(analysis_path, "w", encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=4, ensure_ascii=False)
            
            logger.info("Translation analysis success.")
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            import traceback
            traceback.print_exc()

        # 5. Final Rendering
        if analysis_data:
            logger.info(">>> Step 4: Final Rendering")
            renderer = TextRenderer()
            
            # Decide which background to use
            bg_to_use = inpainted_path if (os.path.exists(inpainted_path) and inpainted_path != processing_image_path) else processing_image_path
            
            # Create outputs directory
            outputs_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            
            # Generate unique filename with timestamp and lang code
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            final_output_path = os.path.join(outputs_dir, f"output_{target_lang}_{timestamp}.jpg")
            
            try:
                renderer.render_text(bg_to_use, analysis_data, final_output_path)
                logger.info(f"Pipeline Completed! Output at: {final_output_path}")

                # [ROI] Merge back if needed
                if roi_coords:
                     logger.info(">>> Step 5: ROI Merging")
                     # Paste processed ROI back to original
                     merged_output_path = final_output_path.replace(".jpg", "_merged.jpg")
                     ImageUtils.merge_image(
                         original_image_path, 
                         final_output_path, 
                         roi_coords[0], 
                         roi_coords[1], 
                         merged_output_path
                     )
                     logger.info(f"ROI Merged Output: {merged_output_path}")

            except Exception as e:
                logger.error(f"Rendering failed: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.critical(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()