# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license='MIT License'
#   Author      : haibingshuai 
#   Created date: 2019/11/8 9:44
#   Description :
# ================================================================

from easydict import EasyDict as edict
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
config = edict()

config.AngleModelPb = os.path.join(dir_path, 'models/Angle-model.pb')
config.AngleModelPbtxt = os.path.join(dir_path, 'models/Angle-model.pbtxt')
config.yoloCfg = os.path.join(dir_path, 'models/text.cfg')
config.yoloWeights = os.path.join(dir_path, 'models/text.weights')

# opencv(cpu) or tf(gpu)
config.AngleModelFlag = 'tf'
