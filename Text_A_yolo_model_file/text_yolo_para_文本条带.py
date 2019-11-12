# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license     : MIT License
#   Author      : haibingshuai 
#   Created date: 2019/11/8 11:18
#   Description :
# ================================================================
from Text_A_yolo_model_file.common_yolo.detectors import TextDetector
from Text_A_yolo_model_file.config import config
from Text_A_yolo_model_file.yolo.keras_detect import text_detect
from Text_A_yolo_model_file.component.image import sort_box
import cv2
import numpy as np
from PIL import Image
from Text_A_yolo_model_file.component.image import rotate_cut_img, draw_bbox

# config 中取值---------------
IMGSIZE = config.IMGSIZE
scale, maxScale = IMGSIZE[0], 2048
MAX_HORIZONTAL_GAP = config.MAX_HORIZONTAL_GAP
MIN_V_OVERLAPS = config.MIN_V_OVERLAPS
MIN_SIZE_SIM = config.MIN_SIZE_SIM
TEXT_PROPOSALS_MIN_SCORE = config.TEXT_PROPOSALS_MIN_SCORE
TEXT_PROPOSALS_NMS_THRESH = config.TEXT_PROPOSALS_NMS_THRESH
TEXT_LINE_NMS_THRESH = config.TEXT_LINE_NMS_THRESH
LINE_MIN_SCORE = config.LINE_MIN_SCORE
leftAdjustAlph = config.leftAdjustAlph
rightAdjustAlph = config.rightAdjustAlph
# ---------------------------

# detect_class:
textdetector = TextDetector(MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM)

# detector
img = cv2.imread('./data_test/img.jpeg')
boxes, scores = text_detect(img, scale=600, maxScale=900)
shape = img.shape[:2]
boxes, scores = textdetector.detect(boxes, scores[:, np.newaxis], shape, TEXT_PROPOSALS_MIN_SCORE,
                                    TEXT_PROPOSALS_NMS_THRESH, TEXT_LINE_NMS_THRESH, LINE_MIN_SCORE)
boxes = sort_box(boxes)
im = Image.fromarray(img)
newBoxes_pos = []
print(len(boxes))
for index, box in enumerate(boxes):
    partImg, box, box_pos = rotate_cut_img(im, box, leftAdjustAlph, rightAdjustAlph)
    newBoxes_pos.append(box_pos)

image = draw_bbox(img, newBoxes_pos)
cv2.imshow('hai', image)
cv2.waitKey(0)
cv2.imwrite('./data_test/img_out.jpeg', image)
