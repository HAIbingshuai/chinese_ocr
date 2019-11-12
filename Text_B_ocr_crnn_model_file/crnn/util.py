# coding=utf-8  python3.6
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   license='MIT License'
#   Author      : haibingshuai 
#   Created date: 2019/11/8 9:51
#   Description :
# ================================================================

import numpy as np
from PIL import Image


def resizeNormalize(img, imgH=32):
    scale = img.size[1] * 1.0 / imgH
    w = img.size[0] / scale
    w = int(w)
    img = img.resize((w, imgH), Image.BILINEAR)
    img = (np.array(img) / 255.0 - 0.5) / 0.5
    return img


def strLabelConverter(res, alphabet):
    N = len(res)
    raw = []
    for i in range(N):
        if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
            raw.append(alphabet[res[i] - 1])
    return ''.join(raw)
