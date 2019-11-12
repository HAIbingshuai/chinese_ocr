#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image

from Text_B_ocr_crnn_model_file.component.image import rotate_cut_img,sort_box


class TextOcrModel(object):
    def __init__(self, ocrModel):
        self.ocrModel = ocrModel

    def ocr_batch(self, img, boxes, leftAdjustAlph=0.0, rightAdjustAlph=0.0):
        im = Image.fromarray(img)
        newBoxes = []
        for index, box in enumerate(boxes):
            partImg, box = rotate_cut_img(im, box, leftAdjustAlph, rightAdjustAlph)
            box['img'] = partImg.convert('L')
            newBoxes.append(box)
        res = self.ocrModel(newBoxes)
        return res

    # def detect_box(self,img,scale=600,maxScale=900):
    #     """
    #     detect text angle in [0,90,180,270]
    #     @@img:np.array
    #     """
    #     boxes,scores = self.textModel(img,scale,maxScale)
    #     return boxes,scores

    # def model(self, img, **args):
    #
    #     #detectAngle = args.get('detectAngle', False)
    #     # if detectAngle:
    #     #     img, angle = self.detect_angle(img)
    #     # else:
    #     #     angle = 0
    #     angle = 0
    #     scale = args.get('scale', 608)
    #     maxScale = args.get('maxScale', 608)
    #     boxes, scores = self.detect_box(img, scale, maxScale)  ##文字检测
    #     boxes, scores = self.box_cluster(img, boxes, scores, **args)
    #     boxes = sort_box(boxes)
    #     leftAdjustAlph = args.get('leftAdjustAlph', 0)
    #     rightAdjustAlph = args.get('rightAdjustAlph', 0)
    #
    #     res = self.ocr_batch(img, boxes, leftAdjustAlph, rightAdjustAlph)
    #     return res, angle
