from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import BasicConvLSTMCell


DISP_SCALING_RESNET50 = 10.0
MIN_DISP = 0.01

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def convLSTM(input, hidden, filters, kernel, scope):

    with tf.variable_scope(scope, initializer = tf.truncated_normal_initializer(stddev=0.1)):
        cell = BasicConvLSTMCell.BasicConvLSTMCell([input.get_shape()[1], input.get_shape()[2]], kernel, filters)

        if hidden is None:
            hidden = cell.zero_state(input.get_shape()[0], tf.float32)
        y_, hidden  = cell(input, hidden)

    return y_, hidden


def rnn_depth_net_decoderlstm(current_input,hidden_state,is_training=True):

    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = current_input.get_shape()[1].value
    W = current_input.get_shape()[2].value

    with tf.variable_scope('rnn_depth_net', reuse = tf.AUTO_REUSE) as sc:

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.005),
                            activation_fn=tf.nn.leaky_relu
                            ):

            cnv1  = slim.conv2d(current_input, 32,  [3, 3], stride=2, scope='cnv1')
            #cnv1b, hidden1 = convLSTM(cnv1, hidden_state[0], 32, [3, 3], scope='cnv1_lstm')
            cnv1b = slim.conv2d(cnv1,  32,  [3, 3], rate=2, stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [3, 3], stride=2, scope='cnv2')
            #cnv2b, hidden2 = convLSTM(cnv2, hidden_state[1], 64, [3, 3], scope='cnv2_lstm')
            cnv2b = slim.conv2d(cnv2,  64,  [3, 3], rate=2, stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            #cnv3b, hidden3 = convLSTM(cnv3, hidden_state[2], 128, [3, 3], scope='cnv3_lstm')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], rate=2, stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            #cnv4b, hidden4 = convLSTM(cnv4, hidden_state[3], 256, [3, 3], scope='cnv4_lstm')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], rate=2, stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 256, [3, 3], stride=2, scope='cnv5')
            #cnv5b, hidden5 = convLSTM(cnv5, hidden_state[4], 256, [3, 3], scope='cnv5_lstm')
            cnv5b = slim.conv2d(cnv5,  256, [3, 3], rate=2, stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 256, [3, 3], stride=2, scope='cnv6')
            #cnv6b, hidden6 = convLSTM(cnv6, hidden_state[5], 256, [3, 3], scope='cnv6_lstm')
            cnv6b = slim.conv2d(cnv6,  256, [3, 3], rate=2, stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            #cnv7b, hidden2 = convLSTM(cnv7, hidden_state[1], 512, [3, 3], scope='cnv7_lstm')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], rate=2, stride=1, scope='cnv7b')


            upcnv7 = slim.conv2d_transpose(cnv7b, 256, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            #icnv7  = slim.conv2d(i7_in, 256, [3, 3], stride=1, scope='icnv7')
            icnv7, hidden7= convLSTM(i7_in, hidden_state[6], 256, [3, 3], scope='icnv7_lstm')

            upcnv6 = slim.conv2d_transpose(icnv7, 128, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            #icnv6  = slim.conv2d(i6_in, 128, [3, 3], stride=1, scope='icnv6')
            icnv6, hidden6= convLSTM(i6_in, hidden_state[5], 128, [3, 3], scope='icnv6_lstm')

            upcnv5 = slim.conv2d_transpose(icnv6, 128, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            #icnv5  = slim.conv2d(i5_in, 128, [3, 3], stride=1, scope='icnv5')
            icnv5, hidden5 = convLSTM(i5_in, hidden_state[4], 128, [3, 3], scope='icnv5_lstm')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            #icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            icnv4, hidden4 = convLSTM(i4_in, hidden_state[3], 128, [3, 3], scope='icnv4_lstm')

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b], axis=3)
            #icnv3  = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
            icnv3, hidden3 = convLSTM(i3_in, hidden_state[2], 64, [3, 3], scope='icnv3_lstm')

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b], axis=3)
            #icnv2  = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
            icnv2, hidden2 = convLSTM(i2_in, hidden_state[1], 32, [3, 3], scope='icnv2_lstm')

            #import pdb;pdb.set_trace()
            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            icnv1, hidden1 = convLSTM(upcnv1, hidden_state[0], 16, [3, 3], scope='icnv1_lstm')
            #icnv1  = slim.conv2d(upcnv1, 16,  [3, 3], stride=1, scope='icnv1')
            depth  = slim.conv2d(icnv1, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1')*DISP_SCALING_RESNET50+MIN_DISP

            return depth, [hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7]#,  hidden8, hidden9,hidden10, hidden11, hidden12, hidden13, hidden14,hidden15, hidden16]


def pose_net(posenet_inputs, hidden_state, is_training=True):
    batch_norm_params = {'is_training': is_training,'decay':0.99}
    with tf.variable_scope('pose_net', reuse = tf.AUTO_REUSE) as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.005),
                            activation_fn=tf.nn.relu):
            conv1  = slim.conv2d(posenet_inputs, 16,  7, 2)
            cnv1b, hidden1 = convLSTM(conv1, hidden_state[0], 16, [3, 3], scope='cnv1_lstm')
            conv2  = slim.conv2d(cnv1b, 32,  5, 2)
            cnv2b, hidden2 = convLSTM(conv2, hidden_state[1], 64, [3, 3], scope='cnv2_lstm')
            conv3  = slim.conv2d(cnv2b, 64,  3, 2)
            cnv3b, hidden3 = convLSTM(conv3, hidden_state[2], 128, [3, 3], scope='cnv3_lstm')
            conv4  = slim.conv2d(cnv3b, 128, 3, 2)
            cnv4b, hidden4 = convLSTM(conv4, hidden_state[3], 256, [3, 3], scope='cnv4_lstm')
            conv5  = slim.conv2d(cnv4b, 256, 3, 2)
            cnv5b, hidden5 = convLSTM(conv5, hidden_state[4], 256, [3, 3], scope='cnv5_lstm')
            conv6  = slim.conv2d(cnv5b, 256, 3, 2)
            cnv6b, hidden6 = convLSTM(conv6, hidden_state[5], 256, [3, 3], scope='cnv6_lstm')
            conv7  = slim.conv2d(cnv6b, 256, 3, 2)
            cnv7b, hidden7 = convLSTM(conv7, hidden_state[6], 512, [3, 3], scope='cnv7_lstm')
            pose_pred = slim.conv2d(cnv7b, 6, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            pose_final = tf.reshape(pose_avg, [-1, 6])*0.01
            return pose_final,[hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7]






def rnn_depth_net_fulllstm(current_input,hidden_state,is_training=True):

    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = current_input.get_shape()[1].value
    W = current_input.get_shape()[2].value

    with tf.variable_scope('rnn_depth_net', reuse = tf.AUTO_REUSE) as sc:

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.005),
                            activation_fn=tf.nn.leaky_relu
                            ):

            cnv1  = slim.conv2d(current_input, 32,  [3, 3], stride=2, scope='cnv1')
            cnv1b, hidden1 = convLSTM(cnv1, hidden_state[0], 32, [3, 3], scope='cnv1_lstm')
            #cnv1b = slim.conv2d(cnv1,  32,  [3, 3], rate=2, stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [3, 3], stride=2, scope='cnv2')
            cnv2b, hidden2 = convLSTM(cnv2, hidden_state[1], 64, [3, 3], scope='cnv2_lstm')
            #cnv2b = slim.conv2d(cnv2,  64,  [3, 3], rate=2, stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b, hidden3 = convLSTM(cnv3, hidden_state[2], 128, [3, 3], scope='cnv3_lstm')
            #cnv3b = slim.conv2d(cnv3,  128, [3, 3], rate=2, stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b, hidden4 = convLSTM(cnv4, hidden_state[3], 256, [3, 3], scope='cnv4_lstm')
            #cnv4b = slim.conv2d(cnv4,  256, [3, 3], rate=2, stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 256, [3, 3], stride=2, scope='cnv5')
            cnv5b, hidden5 = convLSTM(cnv5, hidden_state[4], 256, [3, 3], scope='cnv5_lstm')
            #cnv5b = slim.conv2d(cnv5,  256, [3, 3], rate=2, stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 256, [3, 3], stride=2, scope='cnv6')
            cnv6b, hidden6 = convLSTM(cnv6, hidden_state[5], 256, [3, 3], scope='cnv6_lstm')
            #cnv6b = slim.conv2d(cnv6,  256, [3, 3], rate=2, stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b, hidden7 = convLSTM(cnv7, hidden_state[6], 512, [3, 3], scope='cnv7_lstm')
            #cnv7b = slim.conv2d(cnv7,  512, [3, 3], rate=2, stride=1, scope='cnv7b')

            upcnv7 = slim.conv2d_transpose(cnv7b, 256, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            #icnv7  = slim.conv2d(i7_in, 256, [3, 3], stride=1, scope='icnv7')
            icnv7, hidden8= convLSTM(i7_in, hidden_state[7], 256, [3, 3], scope='icnv7_lstm')

            upcnv6 = slim.conv2d_transpose(icnv7, 128, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            #icnv6  = slim.conv2d(i6_in, 128, [3, 3], stride=1, scope='icnv6')
            icnv6, hidden9= convLSTM(i6_in, hidden_state[8], 128, [3, 3], scope='icnv6_lstm')

            upcnv5 = slim.conv2d_transpose(icnv6, 128, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            #icnv5  = slim.conv2d(i5_in, 128, [3, 3], stride=1, scope='icnv5')
            icnv5, hidden10 = convLSTM(i5_in, hidden_state[9], 128, [3, 3], scope='icnv5_lstm')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            #icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            icnv4, hidden11 = convLSTM(i4_in, hidden_state[10], 128, [3, 3], scope='icnv4_lstm')

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b], axis=3)
            #icnv3  = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
            icnv3, hidden12 = convLSTM(i3_in, hidden_state[11], 64, [3, 3], scope='icnv3_lstm')

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b], axis=3)
            #icnv2  = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
            icnv2, hidden13 = convLSTM(i2_in, hidden_state[12], 32, [3, 3], scope='icnv2_lstm')

            #import pdb;pdb.set_trace()
            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            icnv1, hidden14 = convLSTM(upcnv1, hidden_state[13], 16, [3, 3], scope='icnv1_lstm')
            #icnv1  = slim.conv2d(upcnv1, 16,  [3, 3], stride=1, scope='icnv1')
            depth  = slim.conv2d(icnv1, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1')*DISP_SCALING_RESNET50+MIN_DISP

            return depth, [hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7,  hidden8, hidden9,hidden10, hidden11, hidden12, hidden13, hidden14]



def rnn_depth_net_encoderlstm(current_input,hidden_state,is_training=True):

    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = current_input.get_shape()[1].value
    W = current_input.get_shape()[2].value

    with tf.variable_scope('rnn_depth_net', reuse = tf.AUTO_REUSE) as sc:

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.005),
                            activation_fn=tf.nn.leaky_relu
                            ):

            cnv1  = slim.conv2d(current_input, 32,  [3, 3], stride=2, scope='cnv1')
            cnv1b, hidden1 = convLSTM(cnv1, hidden_state[0], 32, [3, 3], scope='cnv1_lstm')
            #cnv1b = slim.conv2d(cnv1,  32,  [3, 3], rate=2, stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [3, 3], stride=2, scope='cnv2')
            cnv2b, hidden2 = convLSTM(cnv2, hidden_state[1], 64, [3, 3], scope='cnv2_lstm')
            #cnv2b = slim.conv2d(cnv2,  64,  [3, 3], rate=2, stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b, hidden3 = convLSTM(cnv3, hidden_state[2], 128, [3, 3], scope='cnv3_lstm')
            #cnv3b = slim.conv2d(cnv3,  128, [3, 3], rate=2, stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b, hidden4 = convLSTM(cnv4, hidden_state[3], 256, [3, 3], scope='cnv4_lstm')
            #cnv4b = slim.conv2d(cnv4,  256, [3, 3], rate=2, stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 256, [3, 3], stride=2, scope='cnv5')
            cnv5b, hidden5 = convLSTM(cnv5, hidden_state[4], 256, [3, 3], scope='cnv5_lstm')
            #cnv5b = slim.conv2d(cnv5,  256, [3, 3], rate=2, stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 256, [3, 3], stride=2, scope='cnv6')
            cnv6b, hidden6 = convLSTM(cnv6, hidden_state[5], 256, [3, 3], scope='cnv6_lstm')
            #cnv6b = slim.conv2d(cnv6,  256, [3, 3], rate=2, stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b, hidden7 = convLSTM(cnv7, hidden_state[6], 512, [3, 3], scope='cnv7_lstm')
            #cnv7b = slim.conv2d(cnv7,  512, [3, 3], rate=2, stride=1, scope='cnv7b')

            upcnv7 = slim.conv2d_transpose(cnv7b, 256, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 256, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 128, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 128, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 128, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 128, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            upcnv4 = resize_like(upcnv4, cnv3b)
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            upcnv3 = resize_like(upcnv3, cnv2b)
            i3_in  = tf.concat([upcnv3, cnv2b], axis=3)
            icnv3  = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            upcnv2 = resize_like(upcnv2, cnv1b)
            i2_in  = tf.concat([upcnv2, cnv1b], axis=3)
            icnv2  = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            icnv1  = slim.conv2d(upcnv1, 16,  [3, 3], stride=1, scope='icnv1')
            depth  = slim.conv2d(icnv1, 1,   [1, 1], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1')*DISP_SCALING_RESNET50+MIN_DISP # was 10.0

            return depth, [hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7]




def rnn_depth_net_encoderlstm_wpose(current_input,hidden_state,is_training=True):

    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = current_input.get_shape()[1].value
    W = current_input.get_shape()[2].value

    with tf.variable_scope('rnn_depth_net', reuse = tf.AUTO_REUSE) as sc:

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.005),
                            activation_fn=tf.nn.leaky_relu
                            ):

            cnv1  = slim.conv2d(current_input, 32,  [3, 3], stride=2, scope='cnv1')
            cnv1b, hidden1 = convLSTM(cnv1, hidden_state[0], 32, [3, 3], scope='cnv1_lstm')
            #cnv1b = slim.conv2d(cnv1,  32,  [3, 3], rate=2, stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [3, 3], stride=2, scope='cnv2')
            cnv2b, hidden2 = convLSTM(cnv2, hidden_state[1], 64, [3, 3], scope='cnv2_lstm')
            #cnv2b = slim.conv2d(cnv2,  64,  [3, 3], rate=2, stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b, hidden3 = convLSTM(cnv3, hidden_state[2], 128, [3, 3], scope='cnv3_lstm')
            #cnv3b = slim.conv2d(cnv3,  128, [3, 3], rate=2, stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b, hidden4 = convLSTM(cnv4, hidden_state[3], 256, [3, 3], scope='cnv4_lstm')
            #cnv4b = slim.conv2d(cnv4,  256, [3, 3], rate=2, stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 256, [3, 3], stride=2, scope='cnv5')
            cnv5b, hidden5 = convLSTM(cnv5, hidden_state[4], 256, [3, 3], scope='cnv5_lstm')
            #cnv5b = slim.conv2d(cnv5,  256, [3, 3], rate=2, stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 256, [3, 3], stride=2, scope='cnv6')
            cnv6b, hidden6 = convLSTM(cnv6, hidden_state[5], 256, [3, 3], scope='cnv6_lstm')
            #cnv6b = slim.conv2d(cnv6,  256, [3, 3], rate=2, stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b, hidden7 = convLSTM(cnv7, hidden_state[6], 512, [3, 3], scope='cnv7_lstm')
            #cnv7b = slim.conv2d(cnv7,  512, [3, 3], rate=2, stride=1, scope='cnv7b')

            with tf.variable_scope('pose'):

                pose_pred = slim.conv2d(cnv7b, 6, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                pose_final = tf.reshape(pose_avg, [-1, 6])*0.01


            upcnv7 = slim.conv2d_transpose(cnv7b, 256, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 256, [3, 3], stride=1, scope='icnv7')
            #icnv7, hidden8= convLSTM(i7_in, hidden_state[7], 256, [3, 3], scope='icnv7_lstm')

            upcnv6 = slim.conv2d_transpose(icnv7, 128, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 128, [3, 3], stride=1, scope='icnv6')
            #icnv6, hidden9= convLSTM(i6_in, hidden_state[8], 128, [3, 3], scope='icnv6_lstm')

            upcnv5 = slim.conv2d_transpose(icnv6, 128, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 128, [3, 3], stride=1, scope='icnv5')
            #icnv5, hidden10 = convLSTM(i5_in, hidden_state[9], 128, [3, 3], scope='icnv5_lstm')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            upcnv4 = resize_like(upcnv4, cnv3b)
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            #icnv4, hidden11 = convLSTM(i4_in, hidden_state[10], 128, [3, 3], scope='icnv4_lstm')

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            upcnv3 = resize_like(upcnv3, cnv2b)
            i3_in  = tf.concat([upcnv3, cnv2b], axis=3)
            icnv3  = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
            #icnv3, hidden12 = convLSTM(i3_in, hidden_state[11], 64, [3, 3], scope='icnv3_lstm')

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            upcnv2 = resize_like(upcnv2, cnv1b)
            i2_in  = tf.concat([upcnv2, cnv1b], axis=3)
            icnv2  = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
            #icnv2, hidden13 = convLSTM(i2_in, hidden_state[12], 32, [3, 3], scope='icnv2_lstm')

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            #icnv1, hidden14 = convLSTM(upcnv1, hidden_state[13], 16, [3, 3], scope='icnv1_lstm')
            icnv1  = slim.conv2d(upcnv1, 16,  [3, 3], stride=1, scope='icnv1')
            depth  = slim.conv2d(icnv1, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1')*DISP_SCALING_RESNET50+MIN_DISP

            return depth,pose_final, [hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7]







