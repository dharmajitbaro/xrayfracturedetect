import os
import sys
import subprocess
import importlib
import streamlit as st

# --- 1. DETECTRON2 INSTALLATION HACK ---
# We keep the one that we know successfully compiled in the logs!
local_pkg_dir = os.path.join(os.getcwd(), "local_packages")
if local_pkg_dir not in sys.path:
    sys.path.insert(0, local_pkg_dir)

try:
    import detectron2
except ImportError:
    os.makedirs(local_pkg_dir, exist_ok=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "git+https://github.com/facebookresearch/detectron2.git", 
        "--no-deps",               
        "--no-build-isolation",    
        f"--target={local_pkg_dir}" 
    ])
    importlib.invalidate_caches() 

# --- 2. STANDARD IMPORTS ---
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# --- 3. APP SETUP & MODEL LOADING ---
st.set_page_config(page_title="X-ray Fracture Detection", layout="centered")
st.title("🦴 X-Ray Fracture Detection System")
st.write("Upload an X-ray image to detect possible fractures.")

MODEL_PATH = "output_xray/model_final.pth"

@st.cache_resource
def get_predictor():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Could not find model at {MODEL_PATH}. Make sure it uploaded to GitHub!")
        return None
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return DefaultPredictor(cfg)

predictor = get_predictor()

# 🚨 CRUCIAL FIX: Tell the visualizer what class '0' is called!
metadata = MetadataCatalog.get("xray").set(thing_classes=["Fracture"])

# --- 4. MAIN APP LOGIC ---
if predictor:
    uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Resize for CPU stability and speed
        h, w = image.shape[:2]
        if w > 800:
            image = cv2.resize(image, (800, int(h * 800 / w)))
            
        with st.spinner("Analyzing X-ray..."):
            outputs = predictor(image)
            
            # 🚨 CRUCIAL FIX: Define instances so the report can use them
            instances = outputs["instances"].to("cpu")
            num_detections = len(instances)
            
            v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
            out = v.draw_instance_predictions(instances)
            
            st.image(out.get_image()[:, :, ::-1], caption=f"Detected fractures: {num_detections}", use_column_width=True)

        # --- 5. REPORT GENERATION ---
        st.write("### Detection Details")

        report_text = f"X-Ray Fracture Detection Report\n"
        report_text += f"-------------------------------\n"
        report_text += f"Total Fractures Detected: {num_detections}\n"
        
        if num_detections > 0:
            scores = instances.scores.tolist()
            report_text += f"\nConfidence Scores:\n"
            for i, score in enumerate(scores):
                report_text += f"Fracture {i+1}: {score * 100:.2f}%\n"

        st.download_button(
            label="Download Detection Report",
            data=report_text,
            file_name="fracture_report.txt",
            mime="text/plain"
        )
