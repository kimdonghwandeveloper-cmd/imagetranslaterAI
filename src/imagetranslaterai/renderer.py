import os
import requests
import logging
from PIL import Image, ImageDraw, ImageFont # type: ignore
from typing import List, Dict, Any, Tuple
import math
import re

logger = logging.getLogger(__name__)

class TextRenderer:
    def __init__(self, font_path: str = None):
        """
        Args:
            font_path: Path to a TTF/OTF font file. 
                       If None, attempts to download/use a default Korean font.
        """
        self.font_path = font_path or self._get_default_font_path()
        if not os.path.exists(self.font_path):
             self._download_font(self.font_path)

    def _get_default_font_path(self):
        # Default to NanumGothic in assets
        return os.path.join(os.getcwd(), "assets", "NanumGothic.ttf")

    def _download_font(self, save_path):
        logger.info("Downloading default font (NanumGothic)...")
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Bold.ttf"
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(r.content)
            logger.info(f"Font saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to download font: {e}")
            raise

    def render_text(self, bg_image_path: str, analysis_data: List[Dict[str, Any]], output_path: str):
        """
        Renders translated text onto the background image.
        Refactored to use Center Anchor and String Cleaning.
        """
        try:
            # Load Background
            image = Image.open(bg_image_path).convert("RGBA")
            draw = ImageDraw.Draw(image)
            
            # Extract lists for Zip Iteration (simulated if data is already combined, but explicit for clarity)
            ocr_boxes = [item.get('box') for item in analysis_data if 'box' in item]
            translated_texts = [item.get('translated_text', '') for item in analysis_data if 'box' in item]
            styles = [item for item in analysis_data if 'box' in item] # Keep full item for style info
            
            # 1. Pre-calculate optimal font sizes
            sized_items = []
            
            # Use zip as requested to enforce 1:1 pairing
            for box, text, style_item in zip(ocr_boxes, translated_texts, styles):
                # String Cleaning
                clean_text = re.sub(r"[\[\]\'\"]", "", text).strip()
                if not clean_text: continue
                
                size = self._calculate_optimal_font_size(draw, clean_text, box)
                sized_items.append({
                    'box': box, 
                    'text': clean_text, 
                    'style': style_item, 
                    'size': size
                })
                
            # 2. Group and Standardize Sizes
            if sized_items:
                sized_items.sort(key=lambda x: x['size'])
                groups = []
                if sized_items:
                    current_group = [sized_items[0]]
                    for i in range(1, len(sized_items)):
                        curr = sized_items[i]
                        prev = sized_items[i-1]
                        if curr['size'] <= prev['size'] * 1.2:
                            current_group.append(curr)
                        else:
                            groups.append(current_group)
                            current_group = [curr]
                    groups.append(current_group)
                
                for group in groups:
                    sizes = [x['size'] for x in group]
                    target_size = int(sum(sizes) / len(sizes))
                    for entry in group:
                        entry['final_size'] = target_size

            # 3. Draw Text
            for entry in sized_items:
                box = entry['box']
                text = entry['text']
                size = entry.get('final_size', entry['size'])
                style = entry['style']
                
                color_hex = style.get('text_color_hex', '#000000')
                alignment = style.get('alignment', 'center')
                
                self._draw_text(draw, text, box, color_hex, size, alignment)

            final_image = image.convert("RGB")
            final_image.save(output_path)
            logger.info(f"Final image saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            raise

    def _calculate_optimal_font_size(self, draw, text, box) -> int:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        box_width = max(xs) - min(xs)
        box_height = max(ys) - min(ys)
        
        if box_width <= 0 or box_height <= 0: return 10

        font_size = 1
        low, high = 1, int(box_height * 2.0) # check larger range
        optimal_size = 10
        
        while low <= high:
            mid = (low + high) // 2
            font = ImageFont.truetype(self.font_path, mid)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            if text_w <= box_width * 1.2 and text_h <= box_height * 1.2:
                optimal_size = mid
                low = mid + 1
            else:
                high = mid - 1
        return optimal_size

    def _draw_text(self, draw, text, box, color_hex, font_size, alignment):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Center of the box
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        font = ImageFont.truetype(self.font_path, font_size)
        
        # Draw with Middle-Middle anchor
        # NOTE: if alignment is strictly left/right requested, modify anchor x
        
        if alignment == "left":
            draw_x = min_x
            anchor = "lm" # Left-Middle
        elif alignment == "right":
            draw_x = max_x
            anchor = "rm" # Right-Middle
        else:
            draw_x = center_x
            anchor = "mm" # Middle-Middle
            
        draw.text((draw_x, center_y), text, font=font, fill=color_hex, anchor=anchor)
