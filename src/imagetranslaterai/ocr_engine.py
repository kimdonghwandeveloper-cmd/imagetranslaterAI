import logging
from typing import List, Dict, Any
import os
from paddleocr import PaddleOCR # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self, lang: str = 'korean', use_angle_cls: bool = True):
        logger.info(f"Initializing OCR Engine with language='{lang}'")
        try:
            # --- [수정] 박스 감지 파라미터 튜닝 (High-Res & High-Recall) ---
            # det_limit_side_len: 2560 (고해상도 포스터 대응)
            # det_db_box_thresh: 0.4 (흐릿하거나 스타일리시한 폰트 감지)
            # det_db_unclip_ratio: 1.2 (박스 여유 확보)
            self.ocr = PaddleOCR(
                use_angle_cls=use_angle_cls, 
                lang=lang, 
                det_limit_side_len=1280, # Stabilized
                det_db_box_thresh=0.3,   # [Refactor] Increased sensitivity for faint text
                det_db_unclip_ratio=1.05, # [Refactor] Surgical precision (Minimal expansion)
                # ocr_version='PP-OCRv4', # Not supported for Korean yet
                # structure_version='PP-StructureV2'
            ) 
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def detect_text(self, image_input: str) -> List[Dict[str, Any]]:
        # ... (나머지 코드는 기존과 동일하므로 생략, 그대로 쓰시면 됩니다) ...
        # ... detect_text 내부 로직은 기존 코드가 잘 작성되어 있습니다. ...
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
            
        result = self.ocr.ocr(image_input)
        
        # PaddleOCR returns None if no text found
        ocr_data = result[0]
        parsed_results = []
        
        # Check for PaddleX dict structure
        if isinstance(ocr_data, dict):
            # Keys might be 'dt_polys' or 'rec_polys' depending on model/version
            boxes = ocr_data.get('dt_polys') if 'dt_polys' in ocr_data else ocr_data.get('rec_polys')
            texts = ocr_data.get('rec_texts', [])
            scores = ocr_data.get('rec_scores', [])
            
            if boxes is not None:
                for box, text, score in zip(boxes, texts, scores):
                    # Ensure box is a list
                    if hasattr(box, 'tolist'):
                        box = box.tolist()
                    
                    parsed_results.append({
                        'box': box,
                        'text': text,
                        'score': float(score)
                    })
        elif isinstance(ocr_data, list):
             # Fallback for legacy list of lists
             for line in ocr_data:
                valid_line = False
                if isinstance(line, list) and len(line) >= 2:
                    # [[box], (text, score)]
                    box = line[0]
                    content = line[1]
                    if isinstance(content, (tuple, list)) and len(content) >= 2:
                        text = content[0]
                        score = content[1]
                    else:
                        text = str(content)
                        score = 0.99
                    
                    parsed_results.append({
                        'box': box,
                        'text': text,
                        'score': float(score)
                    })

        # Post-Processing: Filtering & Clamping
        # [Filter Logic]
        # 1. Score < 0.6 (Noise)
        # 2. Area < 50px (Tiny dots)
        # 3. Clamping (Out of bounds)
        
        import cv2
        import numpy as np
        
        # Load image to get dimensions
        try:
            img = cv2.imread(image_input)
            if img is None:
                h, w = 99999, 99999 # Safe fallback if image read fails
            else:
                h, w = img.shape[:2]
        except:
            h, w = 99999, 99999

        valid_results = []
        for item in parsed_results:
            box = item['box']
            score = item['score']

            # [Filter 1] Score Check
            if score < 0.6: 
                logger.debug(f"Filtered low score: {score}")
                continue 

            # [Filter 2] Clamping
            clean_box = []
            for point in box:
                # point is [x, y]
                cx = max(0, min(w, point[0]))
                cy = max(0, min(h, point[1]))
                clean_box.append([cx, cy])
            
            # [Filter 3] Area Check
            try:
                poly = np.array(clean_box, dtype=np.int32)
                area = cv2.contourArea(poly)
                if area < 50: 
                    logger.debug(f"Filtered small area: {area}")
                    continue
            except:
                pass # If calculation fails, keep it or skip. Let's keep for safety.

            # Update item
            item['box'] = clean_box
            valid_results.append(item)

        logger.info(f"Detected {len(valid_results)} valid text blocks (Filtered from {len(parsed_results)}).")
        return valid_results
            
        return parsed_results