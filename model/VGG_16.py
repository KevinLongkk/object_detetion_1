########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
# from scipy.misc import imread, imresize
# from imagenet_classes import class_names


class vgg16(object):
    def __init__(self, imgs, pre_per_cell, weights=None):
        # self.imgs = imgs
        self.imgs = imgs * 2.0 - 1.0
        self.pre_per_cell = pre_per_cell
        # self.convlayers()
        # self.fc_layers()
        # self.probs = tf.nn.softmax(self.fc3l)
        # if weights is not None and sess is not None:
        #     self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        # with tf.name_scope('preprocess') as scope:
        #     mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        #     images = self.imgs-mean

        net = slim.conv2d(self.imgs, 3, 7, 2, padding='SAME', scope='conv0')

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')
        # pool1
        # with tf.name_scope('pool_1') as scope:
            # net_1 = slim.conv2d(self.conv1_2, 32, 7, 2, padding='SAME', scope='cas_pool_1_1')
            # net_2 = tf.nn.max_pool(self.conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='cas_pool_1_2')
            # net_2 = tf.transpose(net_2, [0, 1, 3, 2])
            # net_2 = tf.nn.max_pool(net_2, [1, 1, 2, 1], [1, 1, 2, 1], padding='SAME', name='cas_pool_1_3')
            # net_2 = tf.transpose(net_2, [0, 1, 3, 2])
            # self.pool1 = tf.concat([net_1, net_2], axis=-1)

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')
        # # pool2
        # with tf.name_scope('pool_2') as scope:
        #     net_1 = slim.conv2d(self.conv2_2, 64, 7, 2, padding='SAME', scope='cas_pool_2_1')
        #     net_2 = tf.nn.max_pool(self.conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='cas_pool_2_2')
        #     net_2 = tf.transpose(net_2, [0, 1, 3, 2])
        #     net_2 = tf.nn.max_pool(net_2, [1, 1, 2, 1], [1, 1, 2, 1], padding='SAME', name='cas_pool_2_3')
        #     net_2 = tf.transpose(net_2, [0, 1, 3, 2])
        #     self.pool2 = tf.concat([net_1, net_2], axis=-1)

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # with tf.name_scope('pool_3') as scope:
        #     net_1 = slim.conv2d(self.conv3_2, 64, 7, 2, padding='SAME', scope='cas_pool_3_1')
        #     net_2 = tf.nn.max_pool(self.conv3_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='cas_pool_3_2')
        #     net_2 = tf.transpose(net_2, [0, 1, 3, 2])
        #     net_2 = tf.nn.max_pool(net_2, [1, 1, 2, 1], [1, 1, 2, 1], padding='SAME', name='cas_pool_3_3')
        #     net_2 = tf.transpose(net_2, [0, 1, 3, 2])
        #     self.pool3 = tf.concat([net_1, net_2], axis=-1)

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        net = slim.conv2d(self.conv4_3, 512, 3, padding='valid', scope='conv5')

        # FPN_1 = slim.conv2d(net, 128, 3, 2, padding='SAME', scope='FPN_1')

        net_1 = slim.conv2d(net, 256, 9, 2, padding='SAME', scope='conv6_1')
        net_2 = slim.conv2d(net, 256, 5, 2, padding='SAME', scope='conv6_2')
        net_3 = slim.conv2d(net, 256, 3, 2, padding='SAME', scope='conv6_3')
        net_4 = slim.conv2d(net, 256, 2, 2, padding='SAME', scope='conv6_4')
        net = tf.concat([net_1, net_2, net_3, net_4], axis=-1)

        # FPN_2 = slim.conv2d(net, 128, 3, padding='SAME', scope='FPN_2')
        #
        # shortcut = net
        # net = slim.conv2d(net, 1024, 3, padding='SAME', scope='add_conv6_2')
        # net = slim.conv2d(net, 512, 3, padding='SAME', scope='add_conv6_3')
        # net = slim.conv2d(net, 1024, 3, padding='SAME', scope='add_conv6_4')
        # net += shortcut

        # net = slim.conv2d(net, 512, 3, padding='SAME', scope='add_conv_7')

        net_1 = slim.conv2d(net, 64, 7, padding='SAME', scope='conv7_1')
        net_2 = slim.conv2d(net, 64, 5, padding='SAME', scope='conv7_2')
        net_3 = slim.conv2d(net, 64, 3, padding='SAME', scope='conv7_3')
        net_4 = slim.conv2d(net, 64, 2, padding='SAME', scope='conv7_4')
        net = tf.concat([net_1, net_2, net_3, net_4], axis=-1)

        # FPN_3 = slim.conv2d(net, 128, 3, padding='SAME', scope='FPN_3')

        net_1 = slim.conv2d(net, 32, 7, padding='SAME', scope='conv8_1')
        net_2 = slim.conv2d(net, 32, 5, padding='SAME', scope='conv8_2')
        net_3 = slim.conv2d(net, 32, 2, padding='SAME', scope='conv8_3')
        net_4 = slim.conv2d(net, 32, 1, padding='SAME', scope='conv8_4')
        net = tf.concat([net_1, net_2, net_3, net_4], axis=-1)

        # net = net + FPN_1 + FPN_2 + FPN_3

        cls_net = slim.conv2d(net, (20+4) * self.pre_per_cell, 2, padding='SAME', scope='cls_conv1')
        cls_net = slim.conv2d(cls_net, (20+4) * self.pre_per_cell, 2, padding='SAME', scope='cls_conv2')
        cls_net = slim.conv2d(cls_net, (20+4) * self.pre_per_cell, 3, padding='SAME', scope='cls_conv3')
        cls_net = slim.conv2d(cls_net, (20+4) * self.pre_per_cell, 1, padding='SAME', scope='cls_conv4')

        box_net = slim.conv2d(net, 45 * self.pre_per_cell, 2, padding='SAME', scope='box_conv1')
        box_net = slim.conv2d(box_net, 45 * self.pre_per_cell, 1, padding='SAME', scope='box_conv2')

        # net_1 = slim.conv2d(net, 16, 7, padding='SAME', scope='conv8_1')
        # net_2 = slim.conv2d(net, 16, 5, padding='SAME', scope='conv8_2')
        # net_3 = slim.conv2d(net, 16, 2, padding='SAME', scope='conv8_3')
        # net_4 = slim.conv2d(net, 16, 1, padding='SAME', scope='conv8_4')
        # net = tf.concat([net_1, net_2, net_3, net_4], axis=-1)
        #
        # cls_net = slim.conv2d(net, 20, 2, padding='SAME', scope='cls_conv1')
        # cls_net = slim.conv2d(cls_net, 20, 1, padding='SAME', scope='cls_conv2')
        #
        # box_net = slim.conv2d(net, 45, 2, padding='SAME', scope='box_conv1')
        # box_net = slim.conv2d(box_net, 45, 1, padding='SAME', scope='box_conv2')
        # net = slim.dropout(net, self.keep_prob, is_training=self.is_training)
        # net = slim.conv2d(net, 65, 1, padding='SAME', scope='conv35')

        net = tf.concat((box_net, cls_net), axis=-1)
        return net

        # # pool4
        # self.pool4 = tf.nn.max_pool(self.conv4_3,
        #                        ksize=[1, 2, 2, 1],
        #                        strides=[1, 2, 2, 1],
        #                        padding='SAME',
        #                        name='pool4')
        #
        # # conv5_1
        # with tf.name_scope('conv5_1') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv5_1 = tf.nn.relu(out, name=scope)
        #     self.parameters += [kernel, biases]
        #
        # # conv5_2
        # with tf.name_scope('conv5_2') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv5_2 = tf.nn.relu(out, name=scope)
        #     self.parameters += [kernel, biases]
        #
        # # conv5_3
        # with tf.name_scope('conv5_3') as scope:
        #     kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
        #                                              stddev=1e-1), name='weights')
        #     conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        #     biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
        #                          trainable=True, name='biases')
        #     out = tf.nn.bias_add(conv, biases)
        #     self.conv5_3 = tf.nn.relu(out, name=scope)
        #     self.parameters += [kernel, biases]
        #
        # # pool5
        # self.pool5 = tf.nn.max_pool(self.conv5_3,
        #                        ksize=[1, 2, 2, 1],
        #                        strides=[1, 2, 2, 1],
        #                        padding='SAME',
        #                        name='pool4')

    # def fc_layers(self):
    #     # fc1
    #     with tf.name_scope('fc1') as scope:
    #         shape = int(np.prod(self.pool5.get_shape()[1:]))
    #         fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
    #                                                      dtype=tf.float32,
    #                                                      stddev=1e-1), name='weights')
    #         fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
    #                              trainable=True, name='biases')
    #         pool5_flat = tf.reshape(self.pool5, [-1, shape])
    #         fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
    #         self.fc1 = tf.nn.relu(fc1l)
    #         self.parameters += [fc1w, fc1b]
    #
    #     # fc2
    #     with tf.name_scope('fc2') as scope:
    #         fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
    #                                                      dtype=tf.float32,
    #                                                      stddev=1e-1), name='weights')
    #         fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
    #                              trainable=True, name='biases')
    #         fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
    #         self.fc2 = tf.nn.relu(fc2l)
    #         self.parameters += [fc2w, fc2b]
    #
    #     # fc3
    #     with tf.name_scope('fc3') as scope:
    #         fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
    #                                                      dtype=tf.float32,
    #                                                      stddev=1e-1), name='weights')
    #         fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
    #                              trainable=True, name='biases')
    #         self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
    #         self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        '''['conv1_1_W', 'conv1_1_b', 'conv1_2_W', 'conv1_2_b', 'conv2_1_W', 'conv2_1_b', 'conv2_2_W', 'conv2_2_b',
         'conv3_1_W', 'conv3_1_b', 'conv3_2_W', 'conv3_2_b', 'conv3_3_W', 'conv3_3_b', 'conv4_1_W', 'conv4_1_b',
         'conv4_2_W', 'conv4_2_b', 'conv4_3_W', 'conv4_3_b', 'conv5_1_W', 'conv5_1_b', 'conv5_2_W', 'conv5_2_b',
         'conv5_3_W', 'conv5_3_b', 'fc6_W', 'fc6_b', 'fc7_W', 'fc7_b', 'fc8_W', 'fc8_b']'''
        # keys = sorted(weights.keys())
        keys = ['conv1_1_W', 'conv1_1_b', 'conv1_2_W', 'conv1_2_b', 'conv2_1_W', 'conv2_1_b', 'conv2_2_W', 'conv2_2_b',
                'conv3_1_W', 'conv3_1_b', 'conv3_2_W', 'conv3_2_b', 'conv3_3_W', 'conv3_3_b', 'conv4_1_W', 'conv4_1_b',
                'conv4_2_W', 'conv4_2_b', 'conv4_3_W', 'conv4_3_b']
        # print(keys)
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

# if __name__ == '__main__':
#     sess = tf.Session()
#     imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
#     vgg = vgg16(imgs, './weight/vgg16_weights.npz').load_weights('./weight/vgg16_weights.npz', sess)
#
#     img1 = imread('001981.jpg', mode='RGB')
#     img1 = imresize(img1, (224, 224))
#
#     prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
#     preds = (np.argsort(prob)[::-1])[0:5]
#     for p in preds:
#         print(class_names[p], prob[p])