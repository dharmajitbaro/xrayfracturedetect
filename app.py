import streamlit as st
import os
import subprocess
import sys

# --- STEP 1: VERIFIED INSTALLATION ---
@st.cache_resource
def install_detectron2():
    try:
        import detectron2
    except ImportError:
        st.info("📦 Finalizing Detectron2 setup for Python 3.10... (Approx 1 min)")
        # This is a direct link to the pre-compiled Linux wheel for CPU + Python 3.10
        # This bypasses the need for the server to have a C++ compiler
        wheel_url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/detectron2-0.6%2Bcpu-cp310-cp310-linux_x86_64.whl"
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_url])
        st.success("✅ Detectron2 is ready!")
        st.rerun()

install_detectron2()

# --- STEP 2: IMPORTS ---
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# --- STEP 3: APP LOGIC ---
st.title("🦴 X-Ray Fracture Detection System")
MODEL_PATH = "output_xray/model_final.pth"

@st.cache_resource
def get_predictor():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Could not find model at {MODEL_PATH}")
        return None
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # Update if you have more classes
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return DefaultPredictor(cfg)

predictor = get_predictor()

if predictor:
    uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Resize for stability
        h, w = image.shape[:2]
        if w > 800:
            image = cv2.resize(image, (800, int(h * 800 / w)))
            
        with st.spinner("Analyzing..."):
            outputs = predictor(image)
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("xray"), scale=1.0)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            st.image(out.get_image()[:, :, ::-1], use_column_width=True)
