import os
import requests
import logging
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any
import re
import textwrap  # [필수 추가] 줄바꿈 모듈

logger = logging.getLogger(__name__)

class TextRenderer:
    def __init__(self, font_path: str = None):
        self.font_path = font_path or self._get_default_font_path()
        if not os.path.exists(self.font_path):
             self._download_font(self.font_path)

    def _get_default_font_path(self):
        return os.path.join(os.getcwd(), "assets", "NanumGothic.ttf")

    def _download_font(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
        Supports multi-line text wrapping and vertical centering.
        """
        try:
            image = Image.open(bg_image_path).convert("RGBA")
            draw = ImageDraw.Draw(image)
            
            # Filter valid items
            valid_items = [item for item in analysis_data if 'box' in item and 'translated_text' in item]

            for item in valid_items:
                box = item['box']
                text = item['translated_text']
                style = item
                
                # 1. Cleaning
                clean_text = re.sub(r"[\[\]\'\"]", "", str(text)).strip()
                if not clean_text: continue

                # 2. Text Wrapping & Sizing
                font, lines, final_size = self._fit_text_to_box(draw, clean_text, box)
                
                color_hex = style.get('text_color_hex', '#000000')
                
                # 3. Drawing with Vertical Centering
                self._draw_multiline_text(draw, lines, box, color_hex, font, image.width)

            final_image = image.convert("RGB")
            final_image.save(output_path)
            logger.info(f"Final image saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            raise

    def _fit_text_to_box(self, draw, text, box):
        """
        Calculates optimal font size and wraps text to fit within the box.
        Uses binary search for font size.
        """
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        full_w = max(xs) - min(xs)
        full_h = max(ys) - min(ys)
        
        # [수정] Safe Area Logic (5% Internal Padding)
        # 박스를 꽉 채우지 않고 여백을 두어 답답함 방지
        box_w = full_w * 0.95
        box_h = full_h * 0.95
        
        # Min/Max font limits
        min_font_size = 10
        max_font_size = int(box_h * 0.9) 
        if max_font_size < min_font_size: max_font_size = min_font_size
        
        best_font = None
        best_lines = [text]
        best_size = min_font_size

        low, high = min_font_size, max_font_size
        
        while low <= high:
            mid_size = (low + high) // 2
            try:
                font = ImageFont.truetype(self.font_path, mid_size)
            except:
                font = ImageFont.load_default()

            # Estimate wrapping
            avg_char_w = draw.textlength("가", font=font)
            if avg_char_w == 0: avg_char_w = mid_size
            
            wrap_width = max(1, int(box_w / avg_char_w))
            lines = textwrap.wrap(text, width=wrap_width)
            
            # Calculate total height
            # Height = (Line Height * Count)
            # Use specific character '가' or 'A' for reliable height
            bbox = font.getbbox("가")
            line_height = (bbox[3] - bbox[1]) * 1.2 # 1.2 line spacing
            total_text_h = line_height * len(lines)
            
            # Check width of longest line
            max_line_w = 0
            for line in lines:
                w = draw.textlength(line, font=font)
                if w > max_line_w: max_line_w = w

            # Fits? (Allow 10% overflow tolerance for aesthetics)
            if max_line_w <= box_w * 1.1 and total_text_h <= box_h * 1.1:
                best_size = mid_size
                best_font = font
                best_lines = lines
                low = mid_size + 1 # Try larger
            else:
                high = mid_size - 1 # Too big
        
        if best_font is None:
            best_font = ImageFont.truetype(self.font_path, min_font_size)

        return best_font, best_lines, best_size

    def _draw_multiline_text(self, draw, lines, box, color, font, img_w):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        
        # Center of box
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2
        
        # Total block height
        bbox = font.getbbox("가")
        line_height = (bbox[3] - bbox[1]) * 1.2
        total_h = line_height * len(lines)
        
        # Start Y (Vertical Center)
        current_y = center_y - (total_h / 2) + (line_height / 2) 
        
        for line in lines:
            # Horizontal Clamping (inherited concept) + Center Anchor
            # But anchor='mm' handles centering at (x,y).
            # We just need to check if center_x causes clipping?
            # For multiline, clamping is tricky per line if their widths vary.
            # Assuming center_x is generally safe or strictly inside image.
            # If strictly needed, we can clamp center_x based on max_line_w...
            # For now, stick to standard center drawing which is robust enough with wrapping.
            
            draw.text((center_x, current_y), line, font=font, fill=color, anchor="mm")
            current_y += line_height