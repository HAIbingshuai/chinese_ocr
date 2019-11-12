#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import (Conv2D, BatchNormalization, MaxPool2D, Input, Permute, Reshape, Dense, LeakyReLU, Activation,
                          Bidirectional, LSTM, TimeDistributed)
from keras.models import Model
from keras.layers import ZeroPadding2D
from keras.activations import relu
from Text_B_ocr_crnn_model_file.crnn.util import resizeNormalize, strLabelConverter
import numpy as np
import tensorflow as tf
graph = tf.get_default_graph()
def keras_crnn(imgH,nc, nclass, nh, leakyRelu=False, lstmFlag=True):
    data_format = 'channels_first'
    ks = [3, 3, 3, 3, 3, 3, 2]
    ps = [1, 1, 1, 1, 1, 1, 0]
    ss = [1, 1, 1, 1, 1, 1, 1]
    nm = [64, 128, 256, 256, 512, 512, 512]
    imgInput = Input(shape=(1, imgH, None), name='imgInput')

    def convRelu(i, batchNormalization=False, x=None):
        nOut = nm[i]
        if leakyRelu:
            activation = LeakyReLU(alpha=0.2)
        else:
            activation = Activation(relu, name='relu{0}'.format(i))

        x = Conv2D(filters=nOut,
                   kernel_size=ks[i],
                   strides=(ss[i], ss[i]),
                   padding='valid' if ps[i] == 0 else 'same',
                   dilation_rate=(1, 1),
                   activation=None, use_bias=True, data_format=data_format,
                   name='cnn.conv{0}'.format(i)
                   )(x)
        if batchNormalization:
            x = BatchNormalization(epsilon=1e-05, axis=1, momentum=0.1, name='cnn.batchnorm{0}'.format(i))(x)
        x = activation(x)
        return x

    x = imgInput
    x = convRelu(0, batchNormalization=False, x=x)

    x = MaxPool2D(pool_size=(2, 2), name='cnn.pooling{0}'.format(0), padding='valid', data_format=data_format)(x)

    x = convRelu(1, batchNormalization=False, x=x)
    x = MaxPool2D(pool_size=(2, 2), name='cnn.pooling{0}'.format(1), padding='valid', data_format=data_format)(x)

    x = convRelu(2, batchNormalization=True, x=x)
    x = convRelu(3, batchNormalization=False, x=x)
    x = ZeroPadding2D(padding=(0, 1), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='cnn.pooling{0}'.format(2),
                  data_format=data_format)(x)

    x = convRelu(4, batchNormalization=True, x=x)
    x = convRelu(5, batchNormalization=False, x=x)
    x = ZeroPadding2D(padding=(0, 1), data_format=data_format)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='cnn.pooling{0}'.format(3),
                  data_format=data_format)(x)
    x = convRelu(6, batchNormalization=True, x=x)

    x = Permute((3, 2, 1))(x)

    x = Reshape((-1, 512))(x)

    if lstmFlag:
        x = Bidirectional(LSTM(nh, return_sequences=True, use_bias=True,
                               recurrent_activation='sigmoid'))(x)
        x = TimeDistributed(Dense(nh))(x)
        x = Bidirectional(LSTM(nh, return_sequences=True, use_bias=True,
                               recurrent_activation='sigmoid'))(x)
        out = TimeDistributed(Dense(nclass))(x)
    else:
        out = Dense(nclass, name='linear')(x)

    return Model(imgInput, out)


class CRNN:
    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False, lstmFlag=True, alphabet=None):

        self.model = keras_crnn(imgH, nc, nclass, nh, leakyRelu=leakyRelu, lstmFlag=lstmFlag)
        self.alphabet = alphabet

    def load_weights(self, path):
        self.model.load_weights(path)

    def predict(self, image):
        image = resizeNormalize(image, 32)
        image = image.astype(np.float32)
        image = np.array([[image]])
        global graph
        with graph.as_default():
            preds = self.model.predict(image)
        preds = np.argmax(preds, axis=2).reshape((-1,))
        raw = strLabelConverter(preds, self.alphabet)
        return raw

    def predict_job(self, boxes):
        n = len(boxes)
        for i in range(n):
            boxes[i]['text'] = self.predict(boxes[i]['img'])

        return boxes

    def predict_batch(self, boxes, batch_size=1):
        """
        predict on batch
        """

        N = len(boxes)
        res = []
        batch = N // batch_size
        if batch * batch_size != N:
            batch += 1
        for i in range(batch):
            tmpBoxes = boxes[i * batch_size:(i + 1) * batch_size]
            imageBatch = []
            imgW = 0
            for box in tmpBoxes:
                img = box['img']
                image = resizeNormalize(img, 32)
                h, w = image.shape[:2]
                imgW = max(imgW, w)
                imageBatch.append(np.array([image]))

            imageArray = np.zeros((len(imageBatch), 1, 32, imgW), dtype=np.float32)
            n = len(imageArray)
            for j in range(n):
                _, h, w = imageBatch[j].shape
                imageArray[j][:, :, :w] = imageBatch[j]

            global graph
            with graph.as_default():
                preds = self.model.predict(imageArray, batch_size=batch_size)

            preds = preds.argmax(axis=2)
            n = preds.shape[0]
            for j in range(n):
                res.append(strLabelConverter(preds[j,].tolist(), self.alphabet))

        for i in range(N):
            boxes[i]['text'] = res[i]
        return boxes
