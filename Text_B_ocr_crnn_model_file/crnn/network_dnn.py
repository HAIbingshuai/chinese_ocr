#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from Text_B_ocr_crnn_model_file.crnn.util import resizeNormalize, strLabelConverter


class CRNN:
    def __init__(self, alphabet=None):
        self.alphabet = alphabet

    def load_weights(self, path):
        ocrPath = path
        ocrPathtxt = path.replace('.pb', '.pbtxt')
        self.model = cv2.dnn.readNetFromTensorflow(ocrPath, ocrPathtxt)

    def predict(self, image):
        image = resizeNormalize(image, 32)
        image = image.astype(np.float32)
        image = np.array([[image]])
        self.model.setInput(image)
        preds = self.model.forward()
        preds = preds.transpose(0, 2, 3, 1)
        preds = preds[0]
        preds = np.argmax(preds, axis=2).reshape((-1,))
        raw = strLabelConverter(preds, self.alphabet)
        return raw

    def predict_job(self, boxes):
        n = len(boxes)
        for i in range(n):
            boxes[i]['text'] = self.predict(boxes[i]['img'])
        return boxes
