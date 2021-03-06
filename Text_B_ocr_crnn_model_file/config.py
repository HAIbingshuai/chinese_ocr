# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license='MIT License'
#   Author      : haibingshuai 
#   Created date: 2019/11/8 10:01
#   Description :
# ================================================================

from easydict import EasyDict as edict
from Text_B_ocr_crnn_model_file.crnn.chinese_english_keys import *
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
config = edict()

# GPU是否实用
config.GUG = True
# 全中文字典
config.ALPH_CHINESE = alphabetChinese
config.ALPH_ENGLISH = alphabetEnglish

# 中文模型或者纯英文模型
config.ChineseModel = True
# 实用lstm层
config.LstmFlag = True

# 参数设置
config.ocrModelKerasDense = os.path.join(dir_path, 'models/keras_h5/ocr-dense.h5')  # LSTMFLAG = False
config.ocrModelKerasLstm = os.path.join(dir_path, 'models/keras_h5/ocr-lstm.h5')  # LSTMFLAG = True
config.ocrModelKerasEng = os.path.join(dir_path, 'models/keras_h5/ocr-english.h5')  # ChineseModel = False
