# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license     : MIT License
#   Author      : HAIbingshuai 
#   Created date: 2019/11/12 11:32
#   Description :
# ================================================================
from Text_B_ocr_crnn_model_file.text_OCR import text_OCR

if __name__ == '__main__':
    text_ocr = text_OCR()
    text = text_ocr.single_text_ocr('./data_test/hai.jpg')
    print(text)
