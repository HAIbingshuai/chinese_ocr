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

# 配重中/英文model及lstm层-------------------------------
chinse_model = config.ChineseModel
alphabetChinese = config.ALPH_CHINESE
alphabetEnglish = config.ALPH_ENGLISH
LSTMFLAG = config.LstmFlag
if chinse_model:
    if LSTMFLAG:
        ocrModel = config.ocrModelKerasLstm
    else:
        ocrModel = config.ocrModelKerasDense
    alphabet = alphabetChinese
else:
    ocrModel = config.ocrModelKerasEng
    alphabet = config.alphabetEnglish
    LSTMFLAG = True
# ------------------------------------------------------


nclass = len(alphabet) + 1
crnn = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=LSTMFLAG, alphabet=alphabet)

if os.path.exists(ocrModel):
    print('选择的模型为：' + ocrModel)
    crnn.load_weights(ocrModel)
else:
    print("模型路径仔细点!")

##单行识别
img = cv2.imread('./data_test/hai2.jpg')
partImg = Image.fromarray(img)
text = crnn.predict(partImg.convert('L'))
print(text)
