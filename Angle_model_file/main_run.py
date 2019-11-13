# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license     : MIT License
#   Author      : HAIbingshuai 
#   Created date: 2019/11/12 11:48
#   Description :
# ================================================================
import cv2

from Angle_model_file.detect_ANGLE import text_ANGLE

image_path = './data_test/img.jpeg'
image = cv2.imread(image_path)
angle_detect_class = text_ANGLE()
img, angle = angle_detect_class.detect_angle(image)
print(angle)
