import logging
from typing import List, Dict, Any, Union
import os
from paddleocr import PaddleOCR # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self, lang: str = 'korean', use_angle_cls: bool = True):
        """
        Initialize the PaddleOCR engine.
        
        Args:
            lang (str): Language code (e.g., 'korean', 'en', 'ch'). Defaults to 'korean' (supports KR+EN).
            use_angle_cls (bool): Whether to enable text direction classification.
        """
        logger.info(f"Initializing OCR Engine with language='{lang}', use_angle_cls={use_angle_cls}")
        try:
            # Initialize PaddleOCR
            # show_log=False suppresses internal paddle logs to keep stdout clean
            self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang) 
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def detect_text(self, image_input: str) -> List[Dict[str, Any]]:
        """
        Detect text in an image.

        Args:
            image_input (str): Path to the image file.

        Returns:
            List[Dict[str, Any]]: A list of detected text blocks, each containing:
                - 'box': List of [x, y] coordinates for the bounding box.
                - 'text': The detected text string.
                - 'score': Confidence score (float).
        """
        if not os.path.exists(image_input):
            error_msg = f"Image file not found: {image_input}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Running OCR on: {image_input}")
        
        try:
            # PaddleOCR returns a list of results. 
            # Structure: [ [ [box], [text, score] ], ... ]
            # For a single image, result is result[0]
            result = self.ocr.ocr(image_input)
            
            parsed_results = []
            
            
            
            
            # PaddleOCR/PaddleX returns a list containing a dict-like OCRResult
            # Structure: [{'rec_polys': [numpy_array, ...], 'rec_texts': ['text', ...], 'rec_scores': [score, ...]}]
            
            ocr_data = result[0]
            
            # Check if it's the expected dict-like structure (PaddleX)
            if hasattr(ocr_data, '__getitem__') and not isinstance(ocr_data, list):
                try:
                    # Depending on version key might be 'rec_polys' or 'dt_polys'
                    # Based on debug: 'dt_polys' exists and seems correct, 'rec_polys' exists too.
                    # 'rec_texts', 'rec_scores'. 
                    boxes = ocr_data['dt_polys'] if 'dt_polys' in ocr_data else ocr_data.get('rec_polys')
                    texts = ocr_data.get('rec_texts', [])
                    scores = ocr_data.get('rec_scores', [])
                    
                    if not boxes is None:
                        for box, text, score in zip(boxes, texts, scores):
                            # Convert numpy array box to list of lists [[x,y],...]
                            if hasattr(box, 'tolist'):
                                box = box.tolist()
                            
                            parsed_results.append({
                                'box': box,
                                'text': text,
                                'score': float(score)
                            })
                except Exception as e:
                    logger.error(f"Error parsing dict-like result: {e}")
                    
            elif isinstance(ocr_data, list):
                # Legacy list of lists support
                for line in ocr_data:
                    box = line[0]
                    txt = line[1][0]
                    score = line[1][1]
                    parsed_results.append({
                        'box': box,
                        'text': txt,
                        'score': score
                    })
                
            logger.info(f"Detected {len(parsed_results)} text blocks.")
            return parsed_results

        except Exception as e:
            logger.error(f"Error during OCR processing: {e}")
            raise
