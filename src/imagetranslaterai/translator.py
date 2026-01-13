import os
import json
import base64
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, RateLimitError, AuthenticationError

load_dotenv()
logger = logging.getLogger(__name__)

class Translator:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None

        if not self.api_key:
            logger.warning("⚠️ OPENAI_API_KEY not set. Translation will be skipped (fallback to original).")
        else:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.error(f"OpenAI Client Init Failed: {e}")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def translate_and_analyze(self, text_blocks: List[Dict[str, Any]], image_path: str, target_language: str = "Korean") -> List[Dict[str, Any]]:
        """
        Translates text blocks and analyzes style using GPT-4o.
        Supports dynamic target language (e.g., "English", "Japanese").
        """
        # 1. Fallback if no API client
        if not self.client:
            logger.info("No API Key. Returning fallback data.")
            return self._create_fallback_data(text_blocks)

        logger.info(f"Sending request to GPT-4o for translation to {target_language}...")
        
        try:
            base64_image = self._encode_image(image_path)
            
            # Simplify text blocks for the prompt
            blocks_summary = json.dumps([{
                'id': i, 
                'text': b['text'], 
                'box': b['box']
            } for i, b in enumerate(text_blocks)], ensure_ascii=False)

            prompt = f"""
            You are a professional marketing copywriter and visual translator.
            Your goal is to "Localize" text for a **{target_language}** audience while strictly PRESERVING Brand Identity.
            
            Instructions:
            1. **Brand Protection (CRITICAL):** 
               - DO NOT TRANSLATE Brand Names (e.g., 'OEM', 'ODM', 'Canon', 'Nike'). Keep them in English/Original.
               - DO NOT TRANSLATE Specific Product Model Codes (e.g., 'YM-X-3011').
            
            2. **Noise Filtering (CRITICAL):**
               - If a text block seems to be OCR noise (e.g., "1l1.", ";/.", single characters), return empty string "" for `translated_text`.
               - Do not output garbage brackets or symbols.
            
            3. **Summarize & Localize:** 
               - Write concise, natural **{target_language}** marketing copy.
               - Output clean strings only. NO brackets like `['text']`.
            
            Input Data (ID, Text, Box):
            {blocks_summary}
            
            Return ONLY valid JSON in this format:
            [
                {{
                    "id": 0,
                    "translated_text": "Localized Text in {target_language}", 
                    "text_color_hex": "#000000",
                    "alignment": "center" 
                }}
            ]
            """

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
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            # Clean up potential markdown markers
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            analysis_data = json.loads(content)
            
            # Robust ID Matching logic
            ocr_map = {i: block['box'] for i, block in enumerate(text_blocks)}
            
            for item in analysis_data:
                raw_id = item.get('id')
                if raw_id is not None:
                    try:
                        idx = int(raw_id)
                        if idx in ocr_map:
                            item['box'] = ocr_map[idx]
                    except ValueError:
                        logger.warning(f"Invalid ID format from GPT: {raw_id}")
            
            return analysis_data
            
        except RateLimitError:
            logger.error("OpenAI Rate Limit Exceeded. Using fallback.")
            return self._create_fallback_data(text_blocks)

        except AuthenticationError:
            logger.error("OpenAI Authentication Failed. Using fallback.")
            return self._create_fallback_data(text_blocks)

        except Exception as e:
            logger.error(f"GPT-4o translation failed: {e}")
            return self._create_fallback_data(text_blocks)

    def _create_fallback_data(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Helper to create fallback data using original text when API fails.
        """
        fallback_data = []
        for i, block in enumerate(text_blocks):
            fallback_data.append({
                "id": i,
                "translated_text": block['text'],  # Use original text
                "box": block['box'],
                "text_color_hex": "#000000",
                "alignment": "center"
            })
        return fallback_data
