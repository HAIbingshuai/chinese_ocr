# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license     : MIT License
#   Author      : haibingshuai 
#   Created date: 2019/11/8 9:53
#   Description :
# ================================================================
from Text_B_ocr_crnn_model_file.config import config
from Text_B_ocr_crnn_model_file.crnn.network_keras import CRNN
import os
import cv2
from PIL import Image


class text_OCR(object):
    def __init__(self):
        # 配重中/英文model及lstm层-------------------------------
        self.chinse_model = config.ChineseModel
        self.alphabetChinese = config.ALPH_CHINESE
        self.alphabetEnglish = config.ALPH_ENGLISH
        self.LSTMFLAG = config.LstmFlag
        if self.chinse_model:
            if self.LSTMFLAG:
                self.ocrModel = config.ocrModelKerasLstm
            else:
                self.ocrModel = config.ocrModelKerasDense
            self.alphabet = self.alphabetChinese
        else:
            self.ocrModel = config.ocrModelKerasEng
            self.alphabet = config.alphabetEnglish
            self.LSTMFLAG = True
        # ------------------------------------------------------
        self.nclass = len(self.alphabet) + 1
        self.crnn = CRNN(32, 1, self.nclass, 256, leakyRelu=False, lstmFlag=self.LSTMFLAG, alphabet=self.alphabet)

        if os.path.exists(self.ocrModel):
            print('选择的模型为：' + self.ocrModel)
            self.crnn.load_weights(self.ocrModel)
        else:
            print("模型路径仔细点!")

    def single_text_ocr(self, path):
        img = cv2.imread(path)
        partImg = Image.fromarray(img)
        text = self.crnn.predict(partImg.convert('L'))
        return text

    def interface(self, boxes):
        n = len(boxes)
        for i in range(n):
            boxes[i]['text'] = self.crnn.predict(boxes[i]['img'])
        return boxes
