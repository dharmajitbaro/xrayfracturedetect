import streamlit as st
import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# Define your classes - ensure these match your training order!
CLASS_NAMES = ["Fracture"] 

@st.cache_resource
def load_fracture_model():
    cfg = get_cfg()
    # 1. Load the base config (must match what you used for training)
    # If you used Faster R-CNN, use that config here:
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # 2. Update with your specific project details
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
    cfg.MODEL.WEIGHTS = "output_xray/model_final.pth" # Path to your weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Confidence threshold
    
    return DefaultPredictor(cfg), cfg

st.title("X-Ray Fracture Detection")
st.write("Upload an X-ray image to identify potential fractures.")

predictor, cfg = load_fracture_model()

# Setup metadata for the visualizer
metadata = MetadataCatalog.get("fracture_data").set(thing_classes=CLASS_NAMES)

uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Run prediction
    with st.spinner('Analyzing X-ray...'):
        outputs = predictor(image)
    
    # Visualization
    v = Visualizer(image[:, :, ::-1], 
                   metadata=metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW) # BW background highlights detections
    
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Display Results
    st.image(out.get_image()[:, :, ::-1], caption="Prediction Result", use_column_width=True)
    
    # Show raw count
    num_instances = len(outputs["instances"])
    if num_instances > 0:
        st.success(f"Detected {num_instances} potential fracture(s).")
    else:
        st.info("No fractures detected with current confidence threshold.")
