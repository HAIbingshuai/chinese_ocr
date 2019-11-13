# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license='MIT License'
#   Author      : haibingshuai 
#   Created date: 2019/11/8 9:38
#   Description :
# ================================================================

from Angle_model_file.text.opencv_dnn_detect import angle_detect
from PIL import Image
import numpy as np


class text_ANGLE(object):
    def __init__(self):
        self.angleModel = angle_detect

    def detect_angle(self, img):
        """
        detect text angle in [0,90,180,270]
        @@img:np.array
        """
        angle = self.angleModel(img)
        if angle == 90:
            im = Image.fromarray(img).transpose(Image.ROTATE_90)
            img = np.array(im)
        elif angle == 180:
            im = Image.fromarray(img).transpose(Image.ROTATE_180)
            img = np.array(im)
        elif angle == 270:
            im = Image.fromarray(img).transpose(Image.ROTATE_270)
            img = np.array(im)

        return img, angle
