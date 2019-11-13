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
import numpy as np
from PIL import Image
from Text_A_yolo_model_file.component.image import rotate_cut_img, draw_bbox


class text_Yolo(object):
    def __init__(self):
        # config 中取值---------------
        self.IMGSIZE = config.IMGSIZE
        self.scale, self.maxScale = self.IMGSIZE[0], 608
        self.MAX_HORIZONTAL_GAP = config.MAX_HORIZONTAL_GAP
        self.MIN_V_OVERLAPS = config.MIN_V_OVERLAPS
        self.MIN_SIZE_SIM = config.MIN_SIZE_SIM
        self.TEXT_PROPOSALS_MIN_SCORE = config.TEXT_PROPOSALS_MIN_SCORE
        self.TEXT_PROPOSALS_NMS_THRESH = config.TEXT_PROPOSALS_NMS_THRESH
        self.TEXT_LINE_NMS_THRESH = config.TEXT_LINE_NMS_THRESH
        self.LINE_MIN_SCORE = config.LINE_MIN_SCORE
        self.leftAdjustAlph = config.leftAdjustAlph
        self.rightAdjustAlph = config.rightAdjustAlph
        # ---------------------------

        # detect_class:
        self.textdetector = TextDetector(self.MAX_HORIZONTAL_GAP, self.MIN_V_OVERLAPS, self.MIN_SIZE_SIM)

    def detector_OCR(self, img):
        boxes, scores = text_detect(img, self.scale, self.maxScale)
        shape = img.shape[:2]
        boxes, scores = self.textdetector.detect(boxes, scores[:, np.newaxis], shape, self.TEXT_PROPOSALS_MIN_SCORE,
                                                 self.TEXT_PROPOSALS_NMS_THRESH, self.TEXT_LINE_NMS_THRESH,
                                                 self.LINE_MIN_SCORE)
        boxes = sort_box(boxes)
        return boxes

    def detector(self, img):
        boxes = self.detector_OCR(img)
        im = Image.fromarray(img)
        newBoxes_pos = []
        print(len(boxes))
        for index, box in enumerate(boxes):
            partImg, box, box_pos = rotate_cut_img(im, box, self.leftAdjustAlph, self.rightAdjustAlph)
            newBoxes_pos.append(box_pos)

        image = draw_bbox(img, newBoxes_pos)
        return image
