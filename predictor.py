import os
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    )

    # 1. Standard Settings
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    # 2. Path Safety: Ensure it finds the weights file relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, "output_xray", "model_final.pth")
    cfg.MODEL.WEIGHTS = weights_path

    # 3. CPU Optimization: Reduce processing resolution
    # Standard is 800. Lowering to 512 or 640 speeds up CPU inference.
    # Note: If your fractures are tiny, you might need to increase this back up if it misses them.
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 512 

    predictor = DefaultPredictor(cfg)
    return predictor
