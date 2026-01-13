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
            # --- [수정] 박스 감지 파라미터 튜닝 (Ultra-Tight) ---
            # det_db_unclip_ratio: 0.65 (글자 윤곽에 밀착)
            # det_db_box_thresh: 0.5 (흐릿한 글자도 감지)
            self.ocr = PaddleOCR(
                use_angle_cls=use_angle_cls, 
                lang=lang, 
                det_db_unclip_ratio=0.65, 
                det_db_box_thresh=0.5,
                # det_db_score_mode='slow' # Not supported in this version
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

        logger.info(f"Detected {len(parsed_results)} text blocks.")
        return parsed_results
            
        return parsed_results