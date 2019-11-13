# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license     : MIT License
#   Author      : haibingshuai 
#   Created date: 2019/11/8 9:53
#   Description :
# ================================================================
from Angle_model_file.detect_ANGLE import text_ANGLE
from Text_A_yolo_model_file.text_YOLO import text_Yolo
from Text_B_ocr_crnn_model_file.text_OCR import text_OCR
from Total_chinese_ocr.config import config
from Text_B_ocr_crnn_model_file.component.image import rotate_cut_img, union_rbox, adjust_box_to_origin
from PIL import Image
import cv2
import numpy as np


class chinese_OCR(object):
    def __init__(self):
        # 获取angle参数
        self.text_angle_class = text_ANGLE()
        self.detectAngle = config.detectAngle
        # text文本行检测
        self.text_yolo_class = text_Yolo()
        # text文本行识别
        self.text_ocr_class = text_OCR()
        self.leftAdjustAlph = config.leftAdjustAlph
        self.rightAdjustAlph = config.rightAdjustAlph

    def detector(self, img, img_box_path):
        if self.detectAngle:
            img, angle = self.text_angle_class.detect_angle(img)
        else:
            angle = 0
        im = Image.fromarray(img)
        boxes = self.text_yolo_class.detector_OCR(img)

        newBoxes = []
        for index, box in enumerate(boxes):
            partImg, box = rotate_cut_img(im, box, self.leftAdjustAlph, self.rightAdjustAlph)
            box['img'] = partImg.convert('L')
            newBoxes.append(box)
        res = self.text_ocr_class.interface(newBoxes)

        result = union_rbox(res, 0.2)
        res = []
        txt = []
        for i, x in enumerate(result):
            res.append({'text': x['text'], 'name': str(i),
                        'box': {'cx': x['cx'], 'cy': x['cy'], 'w': x['w'], 'h': x['h'], 'angle': x['degree']}
                        })
            txt.append(x['text'])
        res, boxe_xylist = adjust_box_to_origin(img, angle, res)  ## 修正box

        for txt_boxs in boxe_xylist:
            points = np.array(txt_boxs, np.int32)
            cv2.polylines(img, [points], True, (0, 0, 255))
        cv2.imwrite(img_box_path, img)
        return img, res
