# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license     : MIT License
#   Author      : HAIbingshuai 
#   Created date: 2019/11/13 13:32
#   Description :
# ================================================================
import cv2

from z_chinese_ocr.chinese_OCR import chinese_OCR

if __name__ == '__main__':
    # 类
    chinese_ocr = chinese_OCR()
    img = cv2.imread('./data_test/1.jpg')
    pic_out = r'./1_out.jpg'
    img, res = chinese_ocr.detector(img, pic_out)
