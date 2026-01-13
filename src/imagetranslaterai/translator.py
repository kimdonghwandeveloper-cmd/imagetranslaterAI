import os
import json
import base64
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

class Translator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")
        self.client = OpenAI(api_key=api_key)

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def translate_and_analyze(self, text_blocks: List[Dict[str, Any]], image_path: str) -> List[Dict[str, Any]]:
        """
        Translates text blocks and analyzes style using GPT-4o.
        
        Args:
            text_blocks: List of OCR results [{'text': '...', 'box': [...]}, ...]
            image_path: Path to the original full image for context.
            
        Returns:
            List of dicts with keys: 'original_text', 'translated_text', 'color_hex', 'font_style'
        """
        logger.info("Sending request to GPT-4o for translation and analysis...")
        
        base64_image = self._encode_image(image_path)
        
        # Simplify text blocks for the prompt
        blocks_summary = json.dumps([{
            'id': i, 
            'text': b['text'], 
            'box': b['box']
        } for i, b in enumerate(text_blocks)], ensure_ascii=False)

        prompt = f"""
        You are a professional Korean marketing copywriter and visual translator.
        Your goal is to "Localize" text for Korean audience while strictly PRESERVING Brand Identity.
        
        Instructions:
        1. **Brand Protection (CRITICAL):** 
           - DO NOT TRANSLATE Brand Names (e.g., 'OEM', 'ODM', 'Canon', 'Nike'). Keep them in English/Original.
           - DO NOT TRANSLATE Specific Product Model Codes (e.g., 'YM-X-3011').
           - If a text block is purely a Brand Name or Product Name, return it UNCHANGED or empty translated_text if you want to keep original text image (but we are wiping, so return English).
        
        2. **Summarize & Localize (Others):** 
           - For descriptions/specs, do not translate word-for-word. Write concise, attractive Korean marketing copy.
           - Use short phrases suitable for posters.
        
        3. **Style Analysis:** Suggest color and alignment.
        
        Input Data (ID, Text, Box):
        {blocks_summary}
        
        Return ONLY valid JSON in this format:
        [
            {{
                "id": 0,
                "original_text": "Original...",
                "translated_text": "Localized or Original(Brand)", 
                "text_color_hex": "#000000",
                "alignment": "left" 
            }}
        ]
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that outputs only valid JSON. Output strictly JSON without markdown."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1 # Lower temperature for stricter adherence to brand rules
            )
            
            content = response.choices[0].message.content.strip()
            # Clean up potential markdown markers if GPT adds them
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            analysis_data = json.loads(content)
            
            # Merge original 'box' data back into the results based on ID
            # This handles cases where GPT might not return the box or modifies it.
            ocr_map = {i: block['box'] for i, block in enumerate(text_blocks)}
            
            for item in analysis_data:
                idx = item.get('id')
                if idx in ocr_map:
                    item['box'] = ocr_map[idx]
                    
            return analysis_data
            
        except Exception as e:
            logger.error(f"GPT-4o translation failed: {e}")
            raise
