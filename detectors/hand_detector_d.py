import torch
import torch.nn as nn
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


class wrist_detect(nn.Module):
    def __init__(self):
        super(wrist_detect, self).__init__()

        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
        self.predictor = DefaultPredictor(cfg)

    def detect_wrist(self,img):
        outputs = self.predictor(img)
        body_keypoint = outputs["instances"].pred_keypoints[0].detach().cpu().numpy()
        body_keypoint = body_keypoint.reshape(-1,17,3)
        result = []
        for body_kps in body_keypoint:
            ## 9,10 = left,right wrist

            left_shoulder, right_shoulder,left_elbow , right_elbow, left_wrist , right_wrist = \
                body_kps[5],body_kps[6],body_kps[7],body_kps[8],body_kps[9],body_kps[10]
            result.append([left_shoulder, left_elbow, left_wrist, right_shoulder, right_elbow,right_wrist])

        return np.array(result)