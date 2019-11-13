# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license='MIT License'
#   Author      : haibingshuai 
#   Created date: 2019/11/8 11:22
#   Description :
# ================================================================
from easydict import EasyDict as edict
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
config = edict()

# yolo3 输入图像尺寸
config.IMGSIZE = (608, 608)

# 模型性能 keras>darknet>opencv
config.yoloTextFlag = 'keras'
config.keras_anchors = '8,11, 8,16, 8,23, 8,33, 8,48, 8,97, 8,139, 8,198, 8,283'
config.class_names = ['none', 'text', ]
config.kerasTextModel = os.path.join(dir_path,'models/keras_h5/text.h5')

#
config.MAX_HORIZONTAL_GAP = 100  ##字符之间的最大间隔，用于文本行的合并
config.MIN_V_OVERLAPS = 0.6
config.MIN_SIZE_SIM = 0.6

config.TEXT_PROPOSALS_MIN_SCORE = 0.1
config.TEXT_PROPOSALS_NMS_THRESH = 0.3
config.TEXT_LINE_NMS_THRESH = 0.99  ##文本行之间测iou值
config.LINE_MIN_SCORE = 0.1
config.leftAdjustAlph = 0.01  ##对检测的文本行进行向左延伸
config.rightAdjustAlph = 0.01  ##对检测的文本行进行向右延伸
