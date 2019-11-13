# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license     : MIT License
#   Author      : HAIbingshuai 
#   Created date: 2019/11/12 14:12
#   Description :
# ================================================================
from Text_A_yolo_model_file.text_YOLO import text_Yolo
import cv2

text_yolo = text_Yolo()

# detector
img = cv2.imread('./data_test/img.jpeg')
image = text_yolo.detector(img)
cv2.imshow('hai', image)
cv2.waitKey(0)
cv2.imwrite('./data_test/img_out.jpeg', image)
