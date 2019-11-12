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
import cv2
import numpy as np

image_path = './data_test/img.jpeg'
image = cv2.imread(image_path)

angle = angle_detect(image)

if angle == 90:
    im = Image.fromarray(image).transpose(Image.ROTATE_90)
    img = np.array(im)

elif angle == 180:
    im = Image.fromarray(image).transpose(Image.ROTATE_180)
    img = np.array(im)

elif angle == 270:
    im = Image.fromarray(image).transpose(Image.ROTATE_270)
    img = np.array(im)
print(angle)




# result = union_rbox(result, 0.2)
# res = [{'text': x['text'],
#         'name': str(i),
#         'box': {'cx': x['cx'],
#                 'cy': x['cy'],
#                 'w': x['w'],
#                 'h': x['h'],
#                 'angle': x['degree']
#
#                 }
#         } for i, x in enumerate(result)]
# res = adjust_box_to_origin(img, angle, res)  ##修正box

