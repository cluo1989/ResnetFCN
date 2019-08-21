# coding: utf-8
'''
The architecture of ResNet is from ry/tensorflow-resnet:
 https://github.com/ry/tensorflow-resnet
The function of FCN is from MarvinTeichmann/tensorflow-fcn:
 https://github.com/MarvinTeichmann/tensorflow-fcn
'''

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
import numpy as np 

import os, time, datetime
from config import Config


MOVING_AVERAGES_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGES_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838]

tf.app.flags.DEFINE_integer('input_size', 224, 'input image size')
activation = tf.nn.relu

def inference(x, is_training):
    pass


def get_deconv_filter(f_shape):
    width = f_shape[0]
    height = f_shape[0]


def upscore_layer(x, shape, num_classes, name, ksize, stride):
    strides = [1, stride, stride, 1]