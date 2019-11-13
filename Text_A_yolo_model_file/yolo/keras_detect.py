#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from config import kerasTextModel, keras_anchors, class_names
from Text_A_yolo_model_file.yolo.keras_yolo3 import yolo_text, box_layer, K
from Text_A_yolo_model_file.component.image import resize_im
from Text_A_yolo_model_file.config import config

from PIL import Image
import numpy as np
import tensorflow as tf

kerasTextModel = config.kerasTextModel
keras_anchors = config.keras_anchors
class_names = config.class_names

graph = tf.get_default_graph()
anchors = [float(x) for x in keras_anchors.split(',')]
anchors = np.array(anchors).reshape(-1, 2)
num_anchors = len(anchors)
num_classes = len(class_names)
textModel = yolo_text(num_classes, anchors)
textModel.load_weights(kerasTextModel)
sess = K.get_session()
image_shape = K.placeholder(shape=(2,))  ##图像原尺寸:h,w
input_shape = K.placeholder(shape=(2,))  ##图像resize尺寸:h,w
box_score = box_layer([*textModel.output, image_shape, input_shape], anchors, num_classes)


def text_detect(img, scale=600, maxScale=900, prob=0.05):
    im = Image.fromarray(img)
    w, h = im.size
    # 短边固定为608,长边max_scale<4000
    w_, h_ = resize_im(w, h, scale=scale, max_scale=2048)

    boxed_image = im.resize((w_, h_), Image.BICUBIC)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    global graph
    with graph.as_default():
        box, scores = sess.run(
            [box_score],
            feed_dict={
                textModel.input: image_data,
                input_shape: [h_, w_],
                image_shape: [h, w],
                K.learning_phase(): 0
            })[0]
    keep = np.where(scores > prob)
    box[:, 0:4][box[:, 0:4] < 0] = 0
    box[:, 0][box[:, 0] >= w] = w - 1
    box[:, 1][box[:, 1] >= h] = h - 1
    box[:, 2][box[:, 2] >= w] = w - 1
    box[:, 3][box[:, 3] >= h] = h - 1
    box = box[keep[0]]
    scores = scores[keep[0]]
    return box, scores
