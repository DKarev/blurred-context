import os
import cv2
import torch, torchvision
import numpy as np

import detectron2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

# load cfg
cfg = model_zoo.get_config("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", trained=True)
cfg.DATASETS.TRAIN = ("COCOstuff_train",)
cfg.DATASETS.TEST = ("COCOstuff_val",)
cfg.DATALOADER.NUM_WORKERS = 4
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 172

#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0049999.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model to visualize confident predictions only

# choose image
#imagefile = "/media/data/philipp_data/COCOstuff/images/val/000000000139.jpg"
imagefile = "/media/data/philipp_data/UnRel_test/images/1007.jpg"
image = cv2.imread(imagefile)

# get predictions for image
predictor = DefaultPredictor(cfg)
outputs = predictor(image)

# visualize predictions
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

os.makedirs("visualizations", exist_ok=True)
outname = imagefile.replace("\"", "").split("/")[-1].split(".")[0] + "_visualization.png"
cv2.imwrite("visualizations/" + outname, out.get_image()[:, :, ::-1])
