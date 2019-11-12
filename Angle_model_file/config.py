# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license='MIT License'
#   Author      : haibingshuai 
#   Created date: 2019/11/8 9:44
#   Description :
# ================================================================

from easydict import EasyDict as edict

config = edict()

config.AngleModelPb = '../Angle_model_file/models/Angle-model.pb'
config.AngleModelPbtxt = '../Angle_model_file/models/Angle-model.pbtxt'
config.yoloCfg = '../Angle_model_file/models/text.cfg'
config.yoloWeights = '../Angle_model_file/models/text.weights'

# opencv(cpu) or tf(gpu)
config.AngleModelFlag = 'tf'
