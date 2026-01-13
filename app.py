import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import os
import shutil
import time

# Custom Modules
from imagetranslaterai.ocr_engine import OCREngine
from imagetranslaterai.translator import Translator
from imagetranslaterai.inpainter import Inpainter
from imagetranslaterai.renderer import TextRenderer
from imagetranslaterai.utils import ImageUtils

# --- Configuration ---
st.set_page_config(layout="wide", page_title="AI Image Translator")

# --- Cached Resource Loading ---
# Load models once to improve performance
@st.cache_resource
def load_pipeline():
    st.write("Loading AI Models... (This happens only once)")
    ocr = OCREngine()
    inpainter = Inpainter()
    translator = Translator()
    renderer = TextRenderer()
    return ocr, inpainter, translator, renderer

try:
    ocr, inpainter, translator, renderer = load_pipeline()
    st.success("AI Models Loaded Successfully!", icon="✅")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()


def main():
    st.title("🖌️ AI Drag-to-Translate Tool")
    st.markdown("Upload an image, draw a box around the text, and translate it!")

    # Sidebar
    st.sidebar.header("Settings")
    target_lang = st.sidebar.selectbox(
        "Target Language", 
        ["Korean", "English", "Japanese", "Chinese", "French", "Spanish"],
        index=2 # Default to Japanese as per recent context
    )
    
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded_file:
        # 1. Save Uploaded File to Disk (Required for path-based pipeline)
        assets_dir = os.path.join(os.getcwd(), "assets")
        os.makedirs(assets_dir, exist_ok=True)
        temp_filename = "temp_web_upload.jpg"
        temp_image_path = os.path.join(assets_dir, temp_filename)
        
        # Save file
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Open for display
        bg_image = Image.open(temp_image_path).convert("RGB")
        
        # Layout: Left (Canvas), Right (Result)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("1. Draw ROI (Rect)")
            
            # Dynamic canvas size
            # Limit max width to avoid scrolling, keep aspect ratio
            max_canvas_width = 700
            canvas_width = min(max_canvas_width, bg_image.width)
            canvas_height = int(canvas_width * bg_image.height / bg_image.width)
            
            # Canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.2)",  # Translucent Orange
                stroke_width=2,
                stroke_color="#FF0000", # Red border
                background_image=bg_image,
                update_streamlit=True,
                drawing_mode="rect",
                key="canvas",
                width=canvas_width,
                height=canvas_height,
            )
            
            st.caption("Tip: Draw one box. The last drawn box will be processed.")

        # 3. Process Logic
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            
            if objects:
                # Get the last drawn object
                obj = objects[-1]
                
                # Scaling coordinates back to original image size
                # Because canvas might be resized effectively in browser vs PIL image
                # st_canvas width/height matches what we passed.
                # So we calculate scale factor.
                scale_x = bg_image.width / canvas_width
                scale_y = bg_image.height / canvas_height
                
                left = int(obj["left"] * scale_x)
                top = int(obj["top"] * scale_y)
                width = int(obj["width"] * scale_x)
                height = int(obj["height"] * scale_y)
                
                # Validation
                if width <= 0 or height <= 0:
                    st.warning("Invalid Selection")
                else:
                    with col2:
                        st.subheader("2. Action")
                        st.info(f"Selected Region: x={left}, y={top}, w={width}, h={height}")
                        
                        if st.button("🚀 Translate Selected Area", type="primary"):
                            status = st.status("Processing...", expanded=True)
                            
                            try:
                                # Step A: Crop
                                status.write("✂️ Cropping ROI...")
                                crop_path = ImageUtils.crop_image(temp_image_path, left, top, width, height)
                                
                                # Step B: Pipeline
                                status.write("🔍 Detecting Text (OCR)...")
                                ocr_results = ocr.detect_text(crop_path)
                                
                                if not ocr_results:
                                    status.update(label="No text detected!", state="error")
                                    st.error("Text not found in the selected area.")
                                else:
                                    # Extract boxes for inpainting
                                    all_boxes = [item['box'] for item in ocr_results]
                                    
                                    status.write("🎨 Inpainting Background...")
                                    mask_path = inpainter.create_mask(crop_path, all_boxes, padding=5)
                                    inpainted_crop_path = os.path.join(assets_dir, "temp_crop_inpainted.webp")
                                    inpainter.inpaint_simple_fill(crop_path, mask_path, inpainted_crop_path)
                                    
                                    status.write(f"🌍 Translating to {target_lang}...")
                                    analysis_data = translator.translate_and_analyze(ocr_results, crop_path, target_language=target_lang)
                                    
                                    status.write("✨ Rendering Text...")
                                    final_crop_output_path = os.path.join(assets_dir, f"temp_crop_output_{target_lang}.jpg")
                                    renderer.render_text(inpainted_crop_path, analysis_data, final_crop_output_path)
                                    
                                    # Step C: Merge
                                    status.write("🔗 Merging back to Original...")
                                    final_merged_path = os.path.join(assets_dir, f"final_merged_output_{int(time.time())}.jpg")
                                    
                                    ImageUtils.merge_image(
                                        temp_image_path,
                                        final_crop_output_path,
                                        left, top,
                                        final_merged_path
                                    )
                                    
                                    status.update(label="Complete!", state="complete", expanded=False)
                                    
                                    # Display Result
                                    st.success("Refactoring Complete!")
                                    st.image(final_merged_path, caption="Final Result", use_column_width=True)
                                    
                            except Exception as e:
                                status.update(label="Error occurred", state="error")
                                st.error(f"Pipeline failed: {e}")
                                import traceback
                                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
