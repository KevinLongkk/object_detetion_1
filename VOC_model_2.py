import tensorflow as tf
from OpenDL.Data import get_pascal_voc_data as data
from tensorflow.contrib import slim
import cv2
import numpy as np
import os
import random
from model import VGG_16
import time


class Model(object):

    def __init__(self, is_training=True):
        """
                VOC 2007
                TRAIN_STATISTICS = {
                    'none': (0, 0),
                    'aeroplane': (238, 306),
                    'bicycle': (243, 353),
                    'bird': (330, 486),
                    'boat': (181, 290),
                    'bottle': (244, 505),
                    'bus': (186, 229),
                    'car': (713, 1250),
                    'cat': (337, 376),
                    'chair': (445, 798),
                    'cow': (141, 259),
                    'diningtable': (200, 215),
                    'dog': (421, 510),
                    'horse': (287, 362),
                    'motorbike': (245, 339),
                    'person': (2008, 4690),
                    'pottedplant': (245, 514),
                    'sheep': (96, 257),
                    'sofa': (229, 248),
                    'train': (261, 297),
                    'tvmonitor': (256, 324),
                    'total': (5011, 12608),
                }
        """
        self.VOC_LABELS = {
            'aeroplane': 0,
            'bicycle': 1,
            'bird': 2,
            'boat': 3,
            'bottle': 4,
            'bus': 5,
            'car': 6,
            'cat': 7,
            'chair': 8,
            'cow': 9,
            'diningtable': 10,
            'dog': 11,
            'horse': 12,
            'motorbike': 13,
            'person': 14,
            'pottedplant': 15,
            'sheep': 16,
            'sofa': 17,
            'train': 18,
            'tvmonitor': 19,
        }

        self.base_class = {
            'Vehicle': (0, (0, 1, 3, 5, 6, 13, 18)),
            'Animal': (1, (2, 7, 9, 11, 12, 16)),
            'Indoor': (2, (4, 8, 10, 15, 17, 19)),
            'Person': (3, (14, )),
        }

        # self.class_weight = {
        #     'aeroplane': 1.4,
        #     'bicycle': 1.2,
        #     'bird': 0.8,
        #     'boat': 1.5,
        #     'bottle': 0.9,
        #     'bus': 2.3,
        #     'car': 0.4,
        #     'cat': 1.0,
        #     'chair': 0.6,
        #     'cow': 1.5,
        #     'diningtable': 1.0, # 2.0
        #     'dog': 0.7,
        #     'horse': 1.0,
        #     'motorbike': 1.3,
        #     'person': 1.0,
        #     'pottedplant': 0.9,
        #     'sheep': 1.5,
        #     'sofa': 1.8,
        #     'train': 1.3,
        #     'tvmonitor': 1.4,
        #     'Vehicle': 0.8,
        #     'Animal': 1.0,
        #     'Indoor': 1.0,
        #     'Person': 1.0
        # }
        self.class_weight = {
            'aeroplane': 1.0,
            'bicycle': 1.4,
            'bird': 1.2,
            'boat': 1.6,
            'bottle': 2.2,
            'bus': 1.2,
            'car': 1.2,
            'cat': 1.1,
            'chair': 1.5,
            'cow': 1.4,
            'diningtable': 0.55,  # 2.0
            'dog': 1.2,
            'horse': 1.3,
            'motorbike': 1.1,
            'person': 1.0,
            'pottedplant': 2.0,
            'sheep': 1.5,
            'sofa': 1.2,
            'train': 1.3,
            'tvmonitor': 1.6,
            'Vehicle': 1.0,
            'Animal': 1.2,
            'Indoor': 1.1,
            'Person': 1.0
        }

        self.new_VOC_LABELS = {
            'aeroplane': (0, 'Vehicle'),
            'bicycle': (1, 'Vehicle'),
            'bird': (2, 'Animal'),
            'boat': (3, 'Vehicle'),
            'bottle': (4, 'Indoor'),
            'bus': (5, 'Vehicle'),
            'car': (6, 'Vehicle'),
            'cat': (7, 'Animal'),
            'chair': (8, 'Indoor'),
            'cow': (9, 'Animal'),
            'diningtable': (10, 'Indoor'),
            'dog': (11, 'Animal'),
            'horse': (12, 'Animal'),
            'motorbike': (13, 'Vehicle'),
            'person': (14, 'Person'),
            'pottedplant': (15, 'Indoor'),
            'sheep': (16, 'Animal'),
            'sofa': (17, 'Indoor'),
            'train': (18, 'Vehicle'),
            'tvmonitor': (19, 'Indoor'),
        }

        self.image_size = 300
        self.batch_size = 32
        self.batch_size_list = [32, 64, 96]
        # self.image_path = '/home/kevin/DataSet/VOCdevkit/VOC2007'
        # self.image_path = '/home/kevin/DataSet/bread'
        # self.image_path = '/home/kevin/DataSet/COCO/VOC_COCO_with_cls'
        self.image_path = '/__DataSet/VOC/VOC2012'
        # self.image_path = '/__DataSet/COCO'
        self.vgg_npz_path = './model/weight/vgg16_weights.npz'
        self.tensorboard_path = './test2/'

        self.num_grids = 9
        self.learning_rate = 1e-3
        # self.save_path = './VOC_fine_tune/'
        self.save_path = '/media/kevin/0E5B14B00E5B14B0/VOC_fine_tune/'

        self.predict_per_cell = 2

        self.num_base_class = 4

        self.COORD_SCALE = 3.0
        self.OBJECT_SCALE = 7.0
        self.NOOBJECT_SCALE = 1.8
        self.CLASS_SCALE = 6.0

        self.num_anchors = 9
        self.num_class = 20
        self.is_training = is_training
        self.keep_prob = 1.0

        self.exclude_node = ['cls_conv1', 'cls_conv2', 'cls_conv3', 'cls_conv4', 'box_conv1', 'box_conv2']

    def VGG_net(self, inputs):
        vgg = VGG_16.vgg16(inputs, self.predict_per_cell, weights='./model/weight/vgg16_weights.npz')
        net = vgg.convlayers()
        return net, vgg

    def networt(self, inputs):
        inputs = inputs * 2. - 1.0
        with slim.arg_scope([slim.conv2d], activation_fn=self.leaky_relu(0.1),
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.001),
                            biases_initializer=tf.constant_initializer(0.1)):
            # inputs += tf.random_normal(tf.shape(inputs), 0.001, 0.0003)
            net = slim.conv2d(inputs, 8, 5, padding='SAME', scope='conv1')

            # net1_1 = slim.conv2d(net, 8, 17, 2, padding='SAME', scope='conv2_1')
            # net1_2 = slim.conv2d(net, 8, 5, 2, padding='SAME', scope='conv2_2')
            # net1_3 = slim.conv2d(net, 8, 3, 2, padding='SAME', scope='conv2_3')
            # net1 = net1_1 + net1_2 + net1_3
            net = slim.max_pool2d(net, 3, 2, padding='SAME', scope='pool1')
            # net = tf.concat((net1, net2), axis=-1)

            net = slim.conv2d(net, 16, 3, padding='SAME', scope='test1')

            shortcut = net
            net = slim.conv2d(net, 16, 3, padding='SAME', scope='conv3')
            net = slim.conv2d(net, 16, 3, padding='SAME', scope='conv4')
            net += shortcut

            # shortcut = net
            # net = slim.conv2d(net, 16, 3, padding='SAME', scope='add_conv4_1')
            # net = slim.conv2d(net, 32, 3, padding='SAME', scope='add_conv4_2')
            # net = slim.conv2d(net, 16, 3, padding='SAME', scope='add_conv4_3')
            # net += shortcut
            # net = slim.max_pool2d(net, 2, 2, padding='SAME', scope='pool2')
            # net = slim.conv2d(net, 16, 3, padding='SAME', scope='conv5')
            # net = slim.conv2d(net, 16, 3, padding='SAME', scope='conv6')

            # net1_1 = slim.conv2d(net, 16, 9, 2, padding='SAME', scope='conv6_2')
            # net1_2 = slim.conv2d(net, 16, 5, 2, padding='SAME', scope='conv6_3')
            # net1_3 = slim.conv2d(net, 16, 3, 2, padding='SAME', scope='conv6_4')
            # net1 = net1_1 + net1_2 + net1_3
            net = slim.max_pool2d(net, 3, 2, padding='SAME', scope='pool3')
            # net = tf.concat((net1, net2), axis=-1)

            net = slim.conv2d(net, 32, 3, padding='SAME', scope='test2')

            shortcut = net
            net = slim.conv2d(net, 32, 3, padding='SAME', scope='conv7')
            net = slim.conv2d(net, 32, 3, padding='SAME', scope='conv8')
            net += shortcut

            shortcut = net
            net = slim.conv2d(net, 32, 3, padding='SAME', scope='add_conv8_1')
            net = slim.conv2d(net, 64, 3, padding='SAME', scope='add_conv8_2')
            net = slim.conv2d(net, 32, 3, padding='SAME', scope='add_conv8_3')
            net += shortcut

            # net1_1 = slim.conv2d(net, 32, 7, 2, padding='SAME', scope='conv9_1')
            # net1_2 = slim.conv2d(net, 32, 5, 2, padding='SAME', scope='conv9_2')
            # net1_3 = slim.conv2d(net, 32, 3, 2, padding='SAME', scope='conv9_3')
            # net1 = net1_1 + net1_2 + net1_3
            net = slim.max_pool2d(net, 3, 2, padding='SAME', scope='pool4')
            # net = tf.concat((net1, net2), axis=-1)

            net = slim.conv2d(net, 64, 3, padding='SAME', scope='test3')

            shortcut = net
            net = slim.conv2d(net, 64, 3, padding='SAME', scope='conv9_4')
            net = slim.conv2d(net, 64, 3, padding='SAME', scope='conv10')
            net += shortcut

            net = slim.conv2d(net, 128, 3, padding='SAME', scope='conv11')

            shortcut = net
            net = slim.conv2d(net, 128, 3, padding='SAME', scope='conv12')
            net = slim.conv2d(net, 128, 3, padding='SAME', scope='conv13')
            net += shortcut

            # net1_1 = slim.conv2d(net, 128, 5, 2, padding='SAME', scope='conv14_1')
            # net1_2 = slim.conv2d(net, 128, 5, 2, padding='SAME', scope='conv14_2')
            # net1_3 = slim.conv2d(net, 128, 3, 2, padding='SAME', scope='conv14_3')
            # net1 = net1_1
            net = slim.max_pool2d(net, 3, 2, padding='SAME', scope='pool5')
            # net = tf.concat((net1, net2), axis=-1)

            net = slim.conv2d(net, 256, 3, padding='SAME', scope='test4')

            shortcut = net
            net = slim.conv2d(net, 256, 3, padding='SAME', scope='conv15')
            net = slim.conv2d(net, 256, 3, padding='SAME', scope='conv16')
            net += shortcut

            shortcut = net
            net = slim.conv2d(net, 256, 1, padding='SAME', scope='add_conv_16_1')
            net = slim.conv2d(net, 256, 3, padding='SAME', scope='add_conv_16_2')
            net = slim.conv2d(net, 256, 1, padding='SAME', scope='add_conv_16_3')
            net += shortcut
            # net1_1 = slim.conv2d(net, 256, 3, 2, padding='SAME', scope='conv16_1')
            # net1 = net1_1
            # net2 = slim.max_pool2d(net, 2, 2, padding='SAME', scope='conv16_2')
            # net = tf.concat((net1, net2), axis=-1)

            net = slim.conv2d(net, 512, 3, padding='SAME', scope='conv17')

            shortcut = net

            sshortcut = net
            net = slim.conv2d(net, 512, 3, padding='SAME', scope='conv18')
            net = slim.conv2d(net, 512, 3, padding='SAME', scope='conv19')
            net += shortcut

            net = slim.conv2d(net, 1024, 3, padding='SAME', scope='conv20')

            shortcut = net
            net = slim.conv2d(net, 1024, 3, padding='SAME', scope='conv21')
            net = slim.conv2d(net, 1024, 3, padding='SAME', scope='conv22')
            net += shortcut
            net = slim.conv2d(net, 512, 1, padding='SAME', scope='conv23')
            net += sshortcut

            shortcut = net
            net = slim.conv2d(net, 512, 3, padding='SAME', scope='add_conv_23_1')
            net = slim.conv2d(net, 1024, 3, padding='SAME', scope='add_conv_23_2')
            net = slim.conv2d(net, 512, 3, padding='SAME', scope='add_conv_23_3')
            net += shortcut

            shortcut = net
            net = slim.conv2d(net, 512, 3, padding='SAME', scope='conv24')
            net = slim.conv2d(net, 512, 1, padding='SAME', scope='conv25')
            net += shortcut

            net = slim.conv2d(net, 256, 3, padding='Valid', scope='conv26')

            shortcut = net
            net = slim.conv2d(net, 256, 1, padding='SAME', scope='conv27')
            net = slim.conv2d(net, 256, 3, padding='SAME', scope='conv28')
            net += shortcut

            net = slim.conv2d(net, 128, 3, padding='SAME', scope='conv29')

            shortcut = net
            net = slim.conv2d(net, 128, 1, padding='SAME', scope='conv30')
            net = slim.conv2d(net, 128, 3, padding='SAME', scope='conv31')
            net += shortcut

            shortcut = net
            net = slim.conv2d(net, 128, 3, padding='SAME', scope='add_conv31_1')
            net = slim.conv2d(net, 128, 1, padding='SAME', scope='add_conv31_2')
            net = slim.conv2d(net, 128, 3, padding='SAME', scope='add_conv31_3')
            net += shortcut

            net = slim.max_pool2d(net, 3, padding='SAME', scope='add_max_pool')

            # mask = slim.conv2d(net, 1, 7, padding='SAME', scope='mask_conv1')
            # mask = slim.conv2d(mask, 1, 3, padding='SAME', scope='mask_conv2', activation_fn=None)
            # mask = tf.sigmoid(mask)
            # mask = mask * 2.0 - 1.0
            # net *= mask
            mask = None

            # net = slim.dropout(net, 0.8, is_training=self.is_training)
            # net = slim.conv2d(net, 64, 1, padding='SAME', scope='conv32')
            # shortcut = net
            # net = slim.conv2d(net, 64, 3, padding='SAME', scope='conv33')
            # net = slim.conv2d(net, 64, 1, padding='SAME', scope='conv34')
            # net += shortcut

            cls_net = slim.conv2d(net, (20 + 4) * self.predict_per_cell, 3, padding='SAME', scope='cls_conv1')
            cls_net = slim.dropout(cls_net, 0.6, is_training=self.is_training)
            cls_net = slim.conv2d(cls_net, (20 + 4) * self.predict_per_cell, 3, padding='SAME', scope='cls_conv2')
            cls_net = slim.conv2d(cls_net, (20 + 4) * self.predict_per_cell, 3, padding='SAME', scope='cls_conv3')
            # cls_net = cls_net * mask
            cls_net_1 = slim.conv2d(cls_net, (20 + 4), 5, padding='SAME', scope='cls_conv4', activation_fn=None)
            cls_net_2 = slim.conv2d(cls_net, (20 + 4), 5, padding='SAME', scope='cls_conv5', activation_fn=None)
            cls_net = tf.concat([cls_net_1, cls_net_2], axis=-1)
            # cls_net = slim.flatten(cls_net, scope='flat_cls')
            # cls_net = slim.dropout(cls_net, 0.9, is_training=self.is_training)
            # cls_net = slim.fully_connected(cls_net, 4096, scope='cls_flc_1')
            # cls_net = slim.fully_connected(cls_net, self.num_grids*self.num_grids*
            #                                self.predict_per_cell*24, scope='cls_flc_2')
            # cls_net = tf.reshape(cls_net, (-1, self.num_grids, self.num_grids, self.predict_per_cell*24), name='r1')

            box_net = slim.conv2d(net, 45 * self.predict_per_cell, 3, padding='SAME', scope='box_conv1')
            box_net = slim.dropout(box_net, 0.6, is_training=self.is_training)
            box_net = slim.conv2d(box_net, 45 * self.predict_per_cell, 3, padding='SAME', scope='add_box_conv1')
            # box_net = box_net * mask
            box_net_1 = slim.conv2d(box_net, 45, 5, padding='SAME', scope='box_conv2', activation_fn=None)
            box_net_2 = slim.conv2d(box_net, 45, 5, padding='SAME', scope='box_conv3', activation_fn=None)
            box_net = tf.concat([box_net_1, box_net_2], axis=-1)
            # box_net = slim.flatten(box_net, scope='flat_box')
            # box_net = slim.dropout(box_net, 0.9, is_training=self.is_training)
            # box_net = slim.fully_connected(box_net, 4096, scope='box_flc_1')
            # box_net = slim.fully_connected(box_net, self.num_grids*self.num_grids*
            #                                self.predict_per_cell*45, scope='box_flc_2')
            # box_net = tf.reshape(box_net, (-1, self.num_grids, self.num_grids, self.predict_per_cell*45), name='r2')

            net = tf.concat((cls_net, box_net), axis=-1)
            return net, mask

    '''
        Anchor boxes
        include three aspect ratio: [1, 1], [1, 2], [2, 1]
        which example as fellow:
        [(50, 50), (50, 100), (100, 50), (75, 75), (75, 150), (150, 75), (100, 100), (100, 200), (200, 100)]

        @:return list
    '''

    def anchor_boxes(self):
        anchor_ratio = [[1, 1], [1, 2], [2, 1]]
        anchor_size = [50 / self.image_size, 75 / self.image_size, 100 / self.image_size]
        anchor_boxes = None
        for size in anchor_size:
            for ratio in anchor_ratio:
                if anchor_boxes is None:
                    anchor_boxes = np.dot(ratio, size)
                else:
                    anchor_boxes = np.append(anchor_boxes, np.dot(ratio, size))
        return tf.constant([[anchor_boxes[x], anchor_boxes[y]] for (x, y) in zip(range(0, 18, 2), range(1, 18, 2))])

    def new_loss(self, labels, predictions, pre_mask):
        anchor_boxes = self.anchor_boxes()
        raw_labels = labels
        raw_predictions = tf.concat((
                tf.reshape(predictions[..., :48], [self.batch_size, self.num_grids, self.num_grids,
                                                   self.predict_per_cell, 24]),
                tf.reshape(predictions[..., 48:], [self.batch_size, self.num_grids, self.num_grids,
                                                   self.predict_per_cell, 45])), axis=-1)
        base_cls_pres = raw_predictions[..., :4]
        cls_pres = raw_predictions[..., 4:24]
        box_pres = raw_predictions[..., 24:]
        labels_boxes = labels[..., :5]
        labels_cls = labels[..., 5:25]
        labels_base_cls = labels[..., 25:]

        labels = tf.tile(labels_boxes, [1, 1, 1, 1, self.num_anchors])
        labels = tf.reshape(labels, (self.batch_size, self.num_grids, self.num_grids, self.predict_per_cell,
                                     self.num_anchors, 5))

        predictions_ = tf.reshape(box_pres, (self.batch_size, self.num_grids, self.num_grids,
                                             self.predict_per_cell, self.num_anchors, 5))
        predictions_ab = tf.stack(
            [predictions_[:, :, :, :, i] * [1., 1., 1., tf.cast(anchor_boxes[i][0], dtype=tf.float32),
             tf.cast(anchor_boxes[i][1], dtype=tf.float32)] for i in range(self.num_anchors)], axis=4)

        iou_predict_truth = self.cal_iou(predictions_ab[..., 1:], labels[..., 1:])

        truth_mask = tf.tile(tf.expand_dims(raw_labels[..., 0], 4), [1, 1, 1, 1, self.num_anchors])

        iou_mask = tf.reduce_max(iou_predict_truth, axis=4, keep_dims=True)
        mask = tf.cast(iou_predict_truth >= tf.minimum(0.95, iou_mask), dtype=tf.float32) * truth_mask

        noobj_mask = 1 - mask
        threshold_noobj_mask = tf.cast(predictions_ab[..., 0] >= 0.05, dtype=tf.float32) * noobj_mask

        coordinate_loss_1 = tf.reduce_mean(tf.reduce_sum((tf.square(tf.sqrt(labels[..., 1:3]) -
                                                          predictions_ab[..., 1:3]) * tf.expand_dims(mask, 5)),
                                                         axis=[1, 2, 3, 4, 5])) * self.COORD_SCALE

        coordinate_loss_1_1 = tf.reduce_mean(tf.reduce_sum(tf.square(predictions_[..., 1]-predictions_[..., 2]) * mask,
                                             axis=[1, 2, 3, 4])) * 0.01

        coordinate_loss_2 = tf.reduce_mean(tf.reduce_sum((tf.square(labels[..., 3:] -
                                                          predictions_ab[..., 3:]) * tf.expand_dims(mask, 5)),
                                                         axis=[1, 2, 3, 4, 5])) * self.COORD_SCALE
        coordinate_loss = coordinate_loss_1 + coordinate_loss_2 + coordinate_loss_1_1

        noobj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(predictions_ab[..., 0]) * threshold_noobj_mask,
                                                  axis=[1, 2, 3, 4])) * self.NOOBJECT_SCALE
        obj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(1.-predictions_ab[..., 0]) * mask,
                                                axis=[1, 2, 3, 4])) * self.OBJECT_SCALE

        # noobj_loss = tf.reduce_mean(tf.reduce_sum(-tf.log(tf.clip_by_value(1-predictions_ab[..., 0], 1e-10, 1.0)) *
        #                                           # tf.square(predictions_ab[..., 0]) *
        #                                           threshold_noobj_mask, axis=[1, 2, 3, 4])) * self.NOOBJECT_SCALE

        # obj_loss = tf.reduce_mean(tf.reduce_sum(-tf.log(tf.clip_by_value(predictions_ab[..., 0], 1e-10, 1.0)) *
        #                                         # tf.square((1 - predictions_ab[..., 0])) * mask,
        #                                         mask,
        #                                         axis=[1, 2, 3, 4])) * self.OBJECT_SCALE

        # cls_loss = tf.reduce_mean(tf.reduce_sum(
        #     tf.nn.softmax_cross_entropy_with_logits(labels=raw_labels[..., 5:25],
        #                                             logits=raw_predictions[..., 45:65]) * raw_labels[..., 0],
        #                                             axis=[1, 2, 3])) * self.CLASS_SCALE

        base_cls_loss = tf.reduce_mean(tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels_base_cls,
                                                    logits=base_cls_pres) * raw_labels[..., 0],
                                                    axis=[1, 2, 3])) * self.CLASS_SCALE * 2.
        cls_loss = tf.reduce_mean(tf.reduce_sum(
            tf.square(labels_cls - cls_pres) * tf.expand_dims(raw_labels[..., 0], 4),
            axis=[1, 2, 3, 4])) * self.CLASS_SCALE


        # base_cls_loss = tf.reduce_mean(tf.reduce_sum(
        #     tf.square(raw_labels[..., 25:] - base_cls_pres) * tf.expand_dims(raw_labels[..., 0], 4),
        #     axis=[1, 2, 3, 4])) * self.CLASS_SCALE * 3

        # obj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(iou_predict_truth-predictions_ab[..., 0]) * mask,
        #                                         axis=[1, 2, 3])) * self.OBJECT_SCALE

        negative_mask = tf.cast(tf.logical_and(predictions_ab < 0., predictions_ab > -1.0),
                                dtype=tf.float32)
        noobj_loss_ = tf.reduce_mean(tf.reduce_sum(-tf.log(1 + predictions_ab * negative_mask),
                                                   axis=[1, 2, 3, 4, 5]))

        # block_label = tf.expand_dims(raw_labels[..., 0, 0], 3)
        # block_obj_mask = block_label
        # block_nobj_mask = 1 - block_obj_mask
        # pre_mask = tf.multiply(pre_mask, 2.0) - 1.0
        # pre_mask_obj_loss = tf.reduce_mean(tf.reduce_sum(-tf.log(tf.clip_by_value(pre_mask, 1e-10, 1.0)) *
        #                                                  block_obj_mask, axis=[1, 2, 3])) * 6.
        # pre_mask_nobj_loss = tf.reduce_mean(tf.reduce_sum(-tf.log(tf.clip_by_value(1.0-pre_mask, 1e-10, 1.0)) *
        #                                                   block_nobj_mask,
        #                                                   axis=[1, 2, 3])) * 0.5

        losses = coordinate_loss + noobj_loss + obj_loss + cls_loss + base_cls_loss + \
                 noobj_loss_

        tf.summary.scalar('coordinate_loss', coordinate_loss)
        tf.summary.scalar('noobj_loss', noobj_loss)
        tf.summary.scalar('obj_loss', obj_loss)
        tf.summary.scalar('cls_loss', cls_loss)
        tf.summary.scalar('base_cls_loss', base_cls_loss)
        # tf.summary.scalar('pre_mask_obj_loss', pre_mask_obj_loss)
        # tf.summary.scalar('pre_mask_nobj_loss', pre_mask_nobj_loss)
        tf.summary.scalar('total_loss', losses)
        # tf.summary.scalar('noobj_loss_', noobj_loss_)

        return losses

    def cal_iou(self, predictions_boxes, labels_boxes):
        # print(predictions_boxes)
        offset_x = tf.constant([x / self.num_grids for x in range(self.num_grids)] * self.num_grids, dtype=tf.float32)
        offset_x = tf.reshape(offset_x, (1, self.num_grids, self.num_grids))
        offset_x = tf.reshape(tf.tile(offset_x, [1, 1, self.predict_per_cell]),
                              (1, self.num_grids, self.num_grids, self.predict_per_cell))
        offset_x = tf.reshape(tf.tile(offset_x, [1, 1, 1, self.num_anchors]),
                              (1, self.num_grids, self.num_grids, self.predict_per_cell, self.num_anchors))

        offset_y = tf.transpose(offset_x, (0, 2, 1, 3, 4))

        labels_offset = tf.stack([
            labels_boxes[..., 0] / self.num_grids + offset_x,
            labels_boxes[..., 1] / self.num_grids + offset_y,
            labels_boxes[..., 2],
            labels_boxes[..., 3]
        ], axis=-1)

        predictions_offset = tf.stack([
            predictions_boxes[..., 0] / self.num_grids + offset_x,
            predictions_boxes[..., 1] / self.num_grids + offset_y,
            predictions_boxes[..., 2],
            predictions_boxes[..., 3]
        ], axis=-1)

        xmin = tf.maximum(labels_offset[..., 0] - labels_offset[..., 2] / 2,
                          predictions_offset[..., 0] - predictions_offset[..., 2] / 2)
        ymin = tf.maximum(labels_offset[..., 1] - labels_offset[..., 3] / 2,
                          predictions_offset[..., 1] - predictions_offset[..., 3] / 2)
        xmax = tf.minimum(labels_offset[..., 0] + labels_offset[..., 2] / 2,
                          predictions_offset[..., 0] + predictions_offset[..., 2] / 2, )
        ymax = tf.minimum(labels_offset[..., 1] + labels_offset[..., 3] / 2,
                          predictions_offset[..., 1] + predictions_offset[..., 3] / 2)
        intersection = tf.maximum(0.0, xmax - xmin) * tf.maximum(0.0, ymax - ymin)
        union = predictions_boxes[..., 2] * predictions_boxes[..., 3] + \
                labels_boxes[..., 2] * labels_boxes[..., 3] - intersection
        union = tf.maximum(union, 1e-10)
        return intersection / union

    def labels_handler(self, labels):
        s_labels = np.zeros((self.batch_size, self.num_grids, self.num_grids,
                             self.labels_handler_num, 25+self.num_base_class), dtype=np.float32)
        for i in range(self.batch_size):
            for label in labels[i]:
                x = (label[1] + label[3]) / 2
                y = (label[2] + label[4]) / 2
                w = label[3] - label[1]
                h = label[4] - label[2]
                for j in range(self.num_grids):
                    if x > j / self.num_grids:
                        x_ind = j
                    if y > j / self.num_grids:
                        y_ind = j
                x_offset, y_offset = (x - x_ind / self.num_grids) * self.num_grids, \
                                     (y - y_ind / self.num_grids) * self.num_grids
                for k in range(self.labels_handler_num):
                    if sum(s_labels[i, x_ind, y_ind, k]) == 0.:
                        s_labels[i, x_ind, y_ind, k, 0] = 1.
                        s_labels[i, x_ind, y_ind, k, 1:5] = x_offset, y_offset, w, h
                        s_labels[i, x_ind, y_ind, k, self.VOC_LABELS[label[0]] + 5] = self.class_weight[label[0]]
                        # s_labels[i, x_ind, y_ind, k, self.VOC_LABELS[label[0]] + 5] = 1.

                        base_class = self.new_VOC_LABELS[label[0]][1]
                        base_class_index = self.base_class[base_class][0]
                        s_labels[i, x_ind, y_ind, k, 25 + base_class_index] = 1.0

                        break
        return s_labels
    
    def labels_handler_bak(self, labels):
        s_labels = np.zeros((self.batch_size, self.num_grids, self.num_grids,
                             self.predict_per_cell, 25+self.num_base_class), dtype=np.float32)
        for i in range(self.batch_size):
            for label in labels[i]:
                x = (label[1] + label[3]) / 2
                y = (label[2] + label[4]) / 2
                w = label[3] - label[1]
                h = label[4] - label[2]
                for j in range(self.num_grids):
                    if x > j / self.num_grids:
                        x_ind = j
                    if y > j / self.num_grids:
                        y_ind = j
                x_offset, y_offset = (x - x_ind / self.num_grids) * self.num_grids, \
                                     (y - y_ind / self.num_grids) * self.num_grids
                for k in range(self.predict_per_cell):
                    if sum(s_labels[i, x_ind, y_ind, k]) == 0.:
                        s_labels[i, x_ind, y_ind, k, 0] = 1.
                        s_labels[i, x_ind, y_ind, k, 1:5] = x_offset, y_offset, w, h
                        s_labels[i, x_ind, y_ind, k, self.VOC_LABELS[label[0]] + 5] = self.class_weight[label[0]]
                        # s_labels[i, x_ind, y_ind, k, self.VOC_LABELS[label[0]] + 5] = 1.

                        base_class = self.new_VOC_LABELS[label[0]][1]
                        base_class_index = self.base_class[base_class][0]
                        s_labels[i, x_ind, y_ind, k, 25 + base_class_index] = 1.0

                        break
        return s_labels

    def train(self):
        print('loading data from :' + self.image_path)
        inputs_ph = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3))
        labels_ph = tf.placeholder(tf.float32, (None, self.num_grids, self.num_grids,
                                                self.predict_per_cell, 25+self.num_base_class))

        tf.summary.image('image', inputs_ph, 3)

        train_data = data.Data(self.image_path, self.batch_size, self.image_size)

        # predictions, vgg = self.VGG_net(inputs_ph)
        predictions, mask = self.networt(inputs_ph)
        loss = self.new_loss(labels_ph, predictions, mask)

        global_step = tf.Variable(0, trainable=False)
        # global_epoch = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 10000, 0.99, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        # grads = optimizer.compute_gradients(loss)
        # for i, (g, v) in enumerate(grads):
        #     if g is not None:
        #         grads[i] = (tf.clip_by_norm(g, 5), v)
        # train_op = optimizer.apply_gradients(grads, global_step)

        s_saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU90%的显存

        with tf.Session(config=config) as sess:
            merged_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.tensorboard_path, sess.graph)
            if len(os.listdir(self.save_path)) != 0:
                sess.run(tf.global_variables_initializer())
                print('restoring from', tf.train.latest_checkpoint(self.save_path))
                try:
                    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2,
                                           reshape=True)
                    saver.restore(sess, tf.train.latest_checkpoint(self.save_path))
                except:
                    variables = tf.contrib.framework.get_variables_to_restore()
                    variables_to_restore = [v for v in variables if v.name.split('/')[0] not in self.exclude_node]
                    init_fn = slim.assign_from_checkpoint_fn(
                        tf.train.latest_checkpoint(self.save_path),
                        variables_to_restore,
                        ignore_missing_vars=True,
                        reshape_variables=True
                    )
                    init_fn(sess)
            else:
                sess.run(tf.global_variables_initializer())
                # if sess.run(global_step) == 0:
                #     print('loading weight from %s' % self.vgg_npz_path)
                #     vgg.load_weights(self.vgg_npz_path, sess)
            sum = 0
            for i in range(1, 100000):
                time_ = time.time()
                images, labels = train_data.load_data(0)
                # images, labels = train_data.load_test_data(0)
                labels = self.labels_handler(labels)
                # self.batch_size = self.batch_size_list[int((random.random() - 1e-10) * 3)]
                read_data_time = time.time() - time_
                # labels = np.ones_like(labels)
                # np.set_printoptions(edgeitems=1000000)
                # print(sess.run(pri[0], feed_dict={inputs_ph: images}))
                # break

                _, step, le_rate, losses = sess.run([optimizer, global_step, learning_rate, loss],
                                           feed_dict={inputs_ph: images, labels_ph: labels})
                # ret_ = sess.run(ret, feed_dict={inputs_ph: images, labels_ph: labels})
                # print(ret_)
                # break
                sum += losses
                totall_time = time.time() - time_
                if i % 10 == 0:
                    llosses = sum / 10.
                    sum = 0
                    print('Global step: %d, 10 steps mean loss is: %f, one step read_data_time: %f, one step totall_time: %f, batch_size: %d, learning_rate: %f'
                          % (step, llosses, read_data_time, totall_time, self.batch_size, le_rate))
                    # global_epoch = global_epoch.assign(train_data.epoch+epoch)
                # if i % 40 == 0:
                    summary_str = sess.run(merged_summary_op, feed_dict={inputs_ph: images, labels_ph: labels})
                    summary_writer.add_summary(summary_str, global_step=step)
                # if i % 100 == 0:
                #     np.set_printoptions(edgeitems=1000000)
                #     print(sess.run(mask[0], feed_dict={inputs_ph: images}))
                if i % 1000 == 0:
                    s_saver.save(sess, self.save_path, global_step=global_step)
                    print('save model success')

    def evaluate(self):

        raw_image = cv2.imread('./image/2007_000027.jpg')
        # raw_image = cv2.imread('/home/kevin/DataSet/COCO/val2017/000000000885.jpg')
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image / 255.
        input = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3))

        # predictions, vgg = self.VGG_net(input)
        predictions = self.networt(input)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('evaluate...')
            print('restoring from', tf.train.latest_checkpoint(self.save_path))
            saver.restore(sess, tf.train.latest_checkpoint(self.save_path))

            raw_predictions = tf.concat((
                tf.reshape(predictions[..., :48], [1, self.num_grids, self.num_grids,
                                                   self.predict_per_cell, 24]),
                tf.reshape(predictions[..., 48:], [1, self.num_grids, self.num_grids,
                                                   self.predict_per_cell, 45])), axis=-1)
            # base_cls_pres = raw_predictions[..., :4]
            # cls_pres = raw_predictions[..., 4:24]
            # box_pres = raw_predictions[..., 24:]

            raw_result = sess.run(raw_predictions, feed_dict={input: [image]})
            # raw_result = np.reshape(raw_result, [1, self.num_grids, self.num_grids,
            #                                      self.predict_per_cell, 65+self.num_base_class])
            result = np.reshape(raw_result[..., 24:], (1, self.num_grids, self.num_grids,
                                                       self.predict_per_cell, self.num_anchors, 5))
            result_cls = raw_result[..., :24]
            for i in range(self.num_grids):
                for j in range(self.num_grids):
                    for m in range(self.predict_per_cell):
                        k = np.where(result[0, i, j, m, :, 0] == np.max(result[0, i, j, m, :, 0]))
                        if len(k[0]) > 1:
                            print(str(i * self.num_grids + j)+'_%d' % m, 0)
                            continue
                        # k = result[0, i, j, :, 0].index(max(result[0, i, j, :, 0]))
                        # if result[0, i, j, 0] > -0.01:
                        if True:
                            # print(result.shape)
                            print(str(i * self.num_grids + j)+'_%d' % m, str(result[0, i, j, m, k, :]))
                            # print(str(result[0, i, j, :]))
            image = self.visual(raw_image, result[0], result_cls[0], threshold=0.3)
            cv2.imshow(' ', image)
            cv2.waitKey(0)

    def evaluate_loop(self):

        # image = cv2.imread('./image/000030.jpg')
        # image_path = '/home/kevin/DataSet/VOCdevkit/VOC_test/VOC2008_test/VOCdevkit/VOC2008/JPEGImages'
        image_path = '/home/kevin/DataSet/VOCdevkit/VOC_test/VOC2012_test/VOC2012/JPEGImages'
        # image_path = '/home/kevin/DataSet/COCO/VOC_COCO_with_cls/JPEGImages'
        # image_path = '/home/kevin/DataSet/bread/JPEGImages'
        # image_path = '/home/kevin/DataSet/VOCdevkit/VOC2012/JPEGImages'

        image_list = os.listdir(image_path)
        # random.shuffle(image_list)
        image_list.sort()

        '''
        test VOC2012
        '''
        # data_path = '/home/kevin/DataSet/VOCdevkit/VOC2012/'
        # with open(os.path.join(data_path, 'ImageSets', 'Main', 'aeroplane_trainval.txt')) as txt:
        #     lines = txt.readlines()
        #     image_list = []
        #     for line in lines:
        #         image_list.append(line.split(' ')[0] + '.jpg')


        input = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3))

        # predictions, vgg = self.VGG_net(input)
        predictions, mask = self.networt(input)
        raw_predictions = tf.concat((
            tf.reshape(predictions[..., :48], [1, self.num_grids, self.num_grids,
                                               self.predict_per_cell, 24]),
            tf.reshape(predictions[..., 48:], [1, self.num_grids, self.num_grids,
                                               self.predict_per_cell, 45])), axis=-1)

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5

        with tf.Session(config=config) as sess:
            print('evaluate...')
            print('restoring from', tf.train.latest_checkpoint(self.save_path))
            saver.restore(sess, tf.train.latest_checkpoint(self.save_path))
            for img_path in image_list:
                print(img_path)
                raw_image = cv2.imread(os.path.join(image_path, img_path))
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.image_size, self.image_size)) / 255.0
                # raw_result, mask_ = sess.run([raw_predictions, mask], feed_dict={input: [image]})
                raw_result = sess.run(raw_predictions, feed_dict={input: [image]})
                # print(image)
                # print(raw_result)

                # raw_result = np.reshape(raw_result, [1, self.num_grids, self.num_grids,
                #                                      self.predict_per_cell, 65+self.num_base_class])
                result = np.reshape(raw_result[..., 24:], (1, self.num_grids, self.num_grids,
                                                           self.predict_per_cell, self.num_anchors, 5))
                result_cls = raw_result[..., :24]
                # for i in range(self.num_grids):
                #     for j in range(self.num_grids):
                #         for m in range(self.predict_per_cell):
                #             k = np.where(result[0, i, j, m, :, 0] == np.max(result[0, i, j, m, :, 0]))
                #             if len(k[0]) > 1:
                #                 print(str(i * self.num_grids + j) + '_%d' % m, 0)
                #                 continue
                            # k = result[0, i, j, :, 0].index(max(result[0, i, j, :, 0]))
                            # if result[0, i, j, 0] > -0.01:
                            # if True:
                            #     # print(result.shape)
                            #     print(str(i * self.num_grids + j) + '_%d' % m, str(result[0, i, j, m, k, :]))
                                # print(str(result[0, i, j, :]))
                image = self.visual(raw_image, result[0], result_cls[0], threshold=0.4)
                cv2.imshow(' ', image)
                # hot_image = self.hot_mask(mask_, raw_image)
                # cv2.imshow('hot_map', hot_image)
                if cv2.waitKey(0) == 27:
                    break

    def webcam(self):
        # cameraCapture = cv2.VideoCapture('./image/test3.mp4')
        cameraCapture = cv2.VideoCapture(0)

        input = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3))
        predictions, mask = self.networt(input)
        raw_predictions = tf.concat((
            tf.reshape(predictions[..., :48], [1, self.num_grids, self.num_grids,
                                               self.predict_per_cell, 24]),
            tf.reshape(predictions[..., 48:], [1, self.num_grids, self.num_grids,
                                               self.predict_per_cell, 45])), axis=-1)
        # videoWriter = cv2.VideoWriter(
        #     'output.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), 30, (300, 300)
        # )
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(self.save_path))
            while True:
                res, raw_frame = cameraCapture.read()
                frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size, self.image_size)) / 255.0
                raw_result = sess.run(raw_predictions, feed_dict={input: [frame]})
                # raw_result = np.reshape(raw_result, [1, self.num_grids, self.num_grids, self.predict_per_cell, 65+4])
                result = np.reshape(raw_result[..., 24:], (1, self.num_grids, self.num_grids,
                                                           self.predict_per_cell, self.num_anchors, 5))
                result_cls = raw_result[..., :24]
                image = self.visual(raw_frame, result[0], result_cls[0], threshold=0.4)
                cv2.imshow(' ', image)
                if cv2.waitKey(20) == 27:
                    break

    def visual(self, image, labels, result_cls, threshold=0.5):
        tf.reset_default_graph()
        with tf.Session() as sess:
            anchor_boxes = sess.run(self.anchor_boxes())
            boxes_1, scores_1, cls_1 = [], [], []
            # boxes_2, scores_2, cls_2 = [], [], []
            width, height = image.shape[1], image.shape[0]
            for m in range(self.predict_per_cell):
                for i in range(self.num_grids):
                    for j in range(self.num_grids):
                        k = np.where(labels[i, j, m, :, 0] == np.max(labels[i, j, m, :, 0]))
                        k = k[0][0]
                        # if i*19+j==104:
                        print(str(i * self.num_grids + j) + '_' + str(k) + ' ' + str(labels[i, j, m, k, 0]))
                        if labels[i, j, m, k, 0] > threshold:

                            labels[i, j, m, k, 1:3] = np.square(labels[i, j, m, k, 1:3])
                            # anchor_boxes[k][0] = np.square(anchor_boxes[k][0])
                            # anchor_boxes[k][1] = np.square(anchor_boxes[k][1])
                            # labels[i, j, m, k, 1:5] /= 100.

                            print(str(i * self.num_grids + j) + '_' + str(m))
                            center_x = (labels[i, j, m, k, 1] + i) / self.num_grids
                            center_y = (labels[i, j, m, k, 2] + j) / self.num_grids
                            xmin, xmax = int((center_x - labels[i, j, m, k, 3] / 2 * anchor_boxes[k][0]) * width), \
                                         int((center_x + labels[i, j, m, k, 3] / 2 * anchor_boxes[k][0]) * width)
                            ymin, ymax = int((center_y - labels[i, j, m, k, 4] / 2 * anchor_boxes[k][1]) * height), \
                                         int((center_y + labels[i, j, m, k, 4] / 2 * anchor_boxes[k][1]) * height)
                            if xmin == xmax:
                                break
                            xmin = 1 if xmin <= 0 else xmin
                            ymin = 1 if ymin <= 0 else ymin
                            xmax = width - 1 if xmax >= width else xmax
                            ymax = height - 1 if ymax >= height else ymax
                            coord = [ymin, xmin, ymax, xmax]

                            boxes_1.append(coord)
                            scores_1.append(labels[i, j, m, k, 0])

                            print(result_cls[i, j, m, :4])
                            print(result_cls[i, j, m, 4:])
                            base_class = sess.run(tf.arg_max(result_cls[i, j, m, :4], -1))
                            base_cls_list, cls_list, cls_key = None, [], None
                            for key in self.base_class.keys():
                                if self.base_class[key][0] == base_class:
                                    base_cls_list = list(self.base_class[key][1])
                                    cls_key = key
                                    break
                            for c in base_cls_list:
                                cls_list.append(result_cls[i, j, m, c+4])
                            cls_index = sess.run(tf.arg_max(cls_list, -1))
                            cls_1.append(self.base_class[cls_key][1][cls_index])

            try:
                truth_boxes_1 = tf.image.non_max_suppression(np.array(boxes_1), np.array(scores_1), 10, 0.45)
                truth_boxes_1 = sess.run(truth_boxes_1)
                # cls = sess.run(cls)

                for i in truth_boxes_1:
                    # r, g, b = random.random(), random.random(), random.random()
                    print(boxes_1[i][1], boxes_1[i][0], boxes_1[i][3], boxes_1[i][2])
                    # cv2.circle(image, (int((boxes_1[i][1]+boxes_1[i][3])/2), int((boxes_1[i][0]+boxes_1[i][2])/2)), 2, (255, 0, 0), 1)
                    cv2.rectangle(image, (boxes_1[i][1], boxes_1[i][0]), (boxes_1[i][3], boxes_1[i][2]), (0, 255, 0), 2)
                    for k in self.VOC_LABELS.keys():
                        if self.VOC_LABELS[k] == cls_1[i]:
                            print(str(k)+str(scores_1[i])[:4])
                            cv2.putText(image, str(k)+str(scores_1[i])[:4], (boxes_1[i][1], boxes_1[i][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                            # cv2.putText(image, str(k), (boxes[i][1], boxes[i][0]),
                            #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            except:
                print('No bbox')
        # for row in range(1, self.num_grids):
        #     r = row * height/self.num_grids
        #     cv2.line(image, (0, int(r)), (width, int(r)), (0, 0, 255), 1)
        # for col in range(1, self.num_grids):
        #     c = col * width/self.num_grids
        #     cv2.line(image, (int(c), 0), (int(c), height), (0, 0, 255), 1)
        # cv2.resize(image, (300, 300))
        tf.get_default_graph().finalize()
        return image

    def visual_bak(self, image, labels, result_cls, threshold=0.5):
        tf.reset_default_graph()
        with tf.Session() as sess:
            anchor_boxes = sess.run(self.anchor_boxes())
            boxes_1, scores_1, cls_1 = [], [], []
            boxes_2, scores_2, cls_2 = [], [], []
            width, height = image.shape[1], image.shape[0]
            for i in range(self.num_grids):
                for j in range(self.num_grids):
                    for m in range(self.predict_per_cell):
                        k = np.where(labels[i, j, m, :, 0] == np.max(labels[i, j, m, :, 0]))
                        k = k[0][0]
                        # if i*19+j==104:
                        print(str(i * self.num_grids + j) + '_' + str(k) + ' ' + str(labels[i, j, m, k, 0]))
                        if labels[i, j, m, k, 0] > threshold:

                            labels[i, j, m, k, 1:3] = np.square(labels[i, j, m, k, 1:3])
                            # anchor_boxes[k][0] = np.square(anchor_boxes[k][0])
                            # anchor_boxes[k][1] = np.square(anchor_boxes[k][1])
                            # labels[i, j, m, k, 1:5] /= 100.

                            print(str(i * self.num_grids + j) + '_' + str(m))
                            center_x = (labels[i, j, m, k, 1] + i) / self.num_grids
                            center_y = (labels[i, j, m, k, 2] + j) / self.num_grids
                            xmin, xmax = int((center_x - labels[i, j, m, k, 3] / 2 * anchor_boxes[k][0]) * width), \
                                         int((center_x + labels[i, j, m, k, 3] / 2 * anchor_boxes[k][0]) * width)
                            ymin, ymax = int((center_y - labels[i, j, m, k, 4] / 2 * anchor_boxes[k][1]) * height), \
                                         int((center_y + labels[i, j, m, k, 4] / 2 * anchor_boxes[k][1]) * height)
                            if xmin == xmax:
                                break
                            xmin = 1 if xmin <= 0 else xmin
                            ymin = 1 if ymin <= 0 else ymin
                            xmax = width - 1 if xmax >= width else xmax
                            ymax = height - 1 if ymax >= height else ymax
                            coord = [ymin, xmin, ymax, xmax]
                            if m == 0:
                                boxes_1.append(coord)
                                scores_1.append(labels[i, j, m, k, 0])

                                print(result_cls[i, j, m, :4])
                                print(result_cls[i, j, m, 4:])
                                base_class = sess.run(tf.arg_max(result_cls[i, j, m, :4], -1))
                                base_cls_list, cls_list, cls_key = None, [], None
                                for key in self.base_class.keys():
                                    if self.base_class[key][0] == base_class:
                                        base_cls_list = list(self.base_class[key][1])
                                        cls_key = key
                                        break
                                for c in base_cls_list:
                                    cls_list.append(result_cls[i, j, m, c+4])
                                cls_index = sess.run(tf.arg_max(cls_list, -1))
                                cls_1.append(self.base_class[cls_key][1][cls_index])
                                # boxes_1[self.base_class[cls_key][1][cls_index]].append(coord)
                                # scores_1[self.base_class[cls_key][1][cls_index]].append(labels[i, j, m, k, 0])
                            elif m == 1:
                                boxes_2.append(coord)
                                scores_2.append(labels[i, j, m, k, 0])

                                print(result_cls[i, j, m, :4])
                                print(result_cls[i, j, m, 4:])
                                base_class = sess.run(tf.arg_max(result_cls[i, j, m, :4], -1))
                                base_cls_list, cls_list, cls_key = None, [], None
                                for key in self.base_class.keys():
                                    if self.base_class[key][0] == base_class:
                                        base_cls_list = list(self.base_class[key][1])
                                        cls_key = key
                                        break
                                for c in base_cls_list:
                                    cls_list.append(result_cls[i, j, m, c+4])
                                cls_index = sess.run(tf.arg_max(cls_list, -1))
                                cls_2.append(self.base_class[cls_key][1][cls_index])
                            # cls.append(sess.run(tf.arg_max(cls_list, -1)))
            try:
                truth_boxes_1 = tf.image.non_max_suppression(np.array(boxes_1), np.array(scores_1), 10, 0.45)
                truth_boxes_1 = sess.run(truth_boxes_1)

                # truth_boxes_1 = []
                # for boxes, scores in zip(boxes_1, scores_1):
                #     truth_boxes_1 += tf.image.non_max_suppression(np.array(boxes), np.array(scores), 10, 0.45)
                # truth_boxes_1 = sess.run(truth_boxes_1)

                # cls = sess.run(cls)

                for i in truth_boxes_1:
                    # r, g, b = random.random(), random.random(), random.random()
                    print(boxes_1[i][1], boxes_1[i][0], boxes_1[i][3], boxes_1[i][2])
                    # cv2.circle(image, (int((boxes_1[i][1]+boxes_1[i][3])/2), int((boxes_1[i][0]+boxes_1[i][2])/2)), 2, (255, 0, 0), 1)
                    cv2.rectangle(image, (boxes_1[i][1], boxes_1[i][0]), (boxes_1[i][3], boxes_1[i][2]), (0, 255, 0), 2)
                    for k in self.VOC_LABELS.keys():
                        if self.VOC_LABELS[k] == cls_1[i]:
                            print(str(k)+str(scores_1[i])[:4])
                            cv2.putText(image, str(k)+str(scores_1[i])[:4], (boxes_1[i][1], boxes_1[i][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                            # cv2.putText(image, str(k), (boxes[i][1], boxes[i][0]),
                            #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                if len(boxes_2) != 0:
                    truth_boxes_2 = tf.image.non_max_suppression(np.array(boxes_2), np.array(scores_2), 10, 0.5)
                    truth_boxes_2 = sess.run(truth_boxes_2)
                    for i in truth_boxes_2:
                        # r, g, b = random.random(), random.random(), random.random()
                        cv2.rectangle(image, (boxes_2[i][1], boxes_2[i][0]), (boxes_2[i][3], boxes_2[i][2]), (0, 255, 0), 1)
                        for k in self.VOC_LABELS.keys():
                            if self.VOC_LABELS[k] == cls_2[i]:
                                print(str(k) + str(scores_2[i])[:4])
                                cv2.putText(image, str(k) + str(scores_2[i])[:4], (boxes_2[i][1], boxes_2[i][0]),
                                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            except:
                print('No bbox')
        # for row in range(1, self.num_grids):
        #     r = row * height/self.num_grids
        #     cv2.line(image, (0, int(r)), (width, int(r)), (0, 0, 255), 1)
        # for col in range(1, self.num_grids):
        #     c = col * width/self.num_grids
        #     cv2.line(image, (int(c), 0), (int(c), height), (0, 0, 255), 1)
        # cv2.resize(image, (300, 300))
        tf.get_default_graph().finalize()
        return image

    # def visual(self, image, labels, result_cls, threshold=0.5):
    #     tf.reset_default_graph()
    #     with tf.Session() as sess:
    #         anchor_boxes = sess.run(self.anchor_boxes())
    #         boxes_1, scores_1, cls_1 = [[] for _ in range(20)], [[] for _ in range(20)], [[] for _ in range(20)]
    #         boxes_2, scores_2, cls_2 = [], [], []
    #         width, height = image.shape[1], image.shape[0]
    #         for i in range(self.num_grids):
    #             for j in range(self.num_grids):
    #                 for m in range(self.predict_per_cell):
    #                     k = np.where(labels[i, j, m, :, 0] == np.max(labels[i, j, m, :, 0]))
    #                     k = k[0][0]
    #                     # if i*19+j==104:
    #                     print(str(i * self.num_grids + j) + '_' + str(k) + ' ' + str(labels[i, j, m, k, 0]))
    #                     if labels[i, j, m, k, 0] > threshold:
    #
    #                         labels[i, j, m, k, 1:3] = np.square(labels[i, j, m, k, 1:3])
    #                         # anchor_boxes[k][0] = np.square(anchor_boxes[k][0])
    #                         # anchor_boxes[k][1] = np.square(anchor_boxes[k][1])
    #                         # labels[i, j, m, k, 1:5] /= 100.
    #
    #                         print(str(i * self.num_grids + j) + '_' + str(m))
    #                         center_x = (labels[i, j, m, k, 1] + i) / self.num_grids
    #                         center_y = (labels[i, j, m, k, 2] + j) / self.num_grids
    #                         xmin, xmax = int((center_x - labels[i, j, m, k, 3] / 2 * anchor_boxes[k][0]) * width), \
    #                                      int((center_x + labels[i, j, m, k, 3] / 2 * anchor_boxes[k][0]) * width)
    #                         ymin, ymax = int((center_y - labels[i, j, m, k, 4] / 2 * anchor_boxes[k][1]) * height), \
    #                                      int((center_y + labels[i, j, m, k, 4] / 2 * anchor_boxes[k][1]) * height)
    #                         if xmin == xmax:
    #                             break
    #                         xmin = 1 if xmin <= 0 else xmin
    #                         ymin = 1 if ymin <= 0 else ymin
    #                         xmax = width - 1 if xmax >= width else xmax
    #                         ymax = height - 1 if ymax >= height else ymax
    #                         coord = [ymin, xmin, ymax, xmax]
    #                         if m == 0:
    #                             # boxes_1.append(coord)
    #                             # scores_1.append(labels[i, j, m, k, 0])
    #
    #                             print(result_cls[i, j, m, :4])
    #                             print(result_cls[i, j, m, 4:])
    #                             base_class = sess.run(tf.arg_max(result_cls[i, j, m, :4], -1))
    #                             base_cls_list, cls_list, cls_key = None, [], None
    #                             for key in self.base_class.keys():
    #                                 if self.base_class[key][0] == base_class:
    #                                     base_cls_list = list(self.base_class[key][1])
    #                                     cls_key = key
    #                                     break
    #                             for c in base_cls_list:
    #                                 cls_list.append(result_cls[i, j, m, c+4])
    #                             cls_index = sess.run(tf.arg_max(cls_list, -1))
    #                             cls_1[self.base_class[cls_key][1][cls_index]].append(self.base_class[cls_key][1][cls_index])
    #                             boxes_1[self.base_class[cls_key][1][cls_index]].append(coord)
    #                             scores_1[self.base_class[cls_key][1][cls_index]].append(labels[i, j, m, k, 0])
    #                         elif m == 1:
    #                             boxes_2.append(coord)
    #                             scores_2.append(labels[i, j, m, k, 0])
    #
    #                             print(result_cls[i, j, m, :4])
    #                             print(result_cls[i, j, m, 4:])
    #                             base_class = sess.run(tf.arg_max(result_cls[i, j, m, :4], -1))
    #                             base_cls_list, cls_list, cls_key = None, [], None
    #                             for key in self.base_class.keys():
    #                                 if self.base_class[key][0] == base_class:
    #                                     base_cls_list = list(self.base_class[key][1])
    #                                     cls_key = key
    #                                     break
    #                             for c in base_cls_list:
    #                                 cls_list.append(result_cls[i, j, m, c+4])
    #                             cls_2.append(self.base_class[cls_key][1][sess.run(tf.arg_max(cls_list, -1))])
    #                         # cls.append(sess.run(tf.arg_max(cls_list, -1)))
    #         # try:
    #             # truth_boxes_1 = tf.image.non_max_suppression(np.array(boxes_1), np.array(scores_1), 10, 0.45)
    #             # truth_boxes_1 = sess.run(truth_boxes_1)
    #
    #         truth_boxes_1 = [[] for _ in range(20)]
    #         for boxes, scores, i in zip(boxes_1, scores_1, range(20)):
    #             if len(boxes) != 0:
    #                 truth_boxes_1[i].append(tf.image.non_max_suppression(np.array(boxes), np.array(scores), 10, 0.45))
    #         truth_boxes_1 = sess.run(truth_boxes_1)
    #         print(truth_boxes_1)
    #
    #
    #         # cls = sess.run(cls)
    #
    #         for j, cls in enumerate(truth_boxes_1):
    #             if len(cls) != 0:
    #                 for i in cls[0]:
    #                     print(boxes_1[j][i])
    #                     # r, g, b = random.random(), random.random(), random.random()
    #                     print(boxes_1[j][i][1], boxes_1[j][i][0], boxes_1[j][i][3], boxes_1[j][i][2])
    #                     # cv2.circle(image, (int((boxes_1[i][1]+boxes_1[i][3])/2), int((boxes_1[i][0]+boxes_1[i][2])/2)), 2, (255, 0, 0), 1)
    #                     cv2.rectangle(image, (boxes_1[j][i][1], boxes_1[j][i][0]), (boxes_1[j][i][3], boxes_1[j][i][2]), (0, 255, 0), 2)
    #                     for k in self.VOC_LABELS.keys():
    #                         if self.VOC_LABELS[k] == cls_1[j][i]:
    #                             print(str(k)+str(scores_1[j][i])[:4])
    #                             cv2.putText(image, str(k)+str(scores_1[j][i])[:4], (boxes_1[j][i][1], boxes_1[j][i][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    #                             # cv2.putText(image, str(k), (boxes[i][1], boxes[i][0]),
    #                             #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    #         if len(boxes_2) != 0:
    #             truth_boxes_2 = tf.image.non_max_suppression(np.array(boxes_2), np.array(scores_2), 10, 0.5)
    #             truth_boxes_2 = sess.run(truth_boxes_2)
    #             for i in truth_boxes_2:
    #                 # r, g, b = random.random(), random.random(), random.random()
    #                 cv2.rectangle(image, (boxes_2[i][1], boxes_2[i][0]), (boxes_2[i][3], boxes_2[i][2]), (0, 255, 0), 1)
    #                 for k in self.VOC_LABELS.keys():
    #                     if self.VOC_LABELS[k] == cls_2[i]:
    #                         print(str(k) + str(scores_2[i])[:4])
    #                         cv2.putText(image, str(k) + str(scores_2[i])[:4], (boxes_2[i][1], boxes_2[i][0]),
    #                                     cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    #         # except:
    #         #     print('No bbox')
    #     # for row in range(1, self.num_grids):
    #     #     r = row * height/self.num_grids
    #     #     cv2.line(image, (0, int(r)), (width, int(r)), (0, 0, 255), 1)
    #     # for col in range(1, self.num_grids):
    #     #     c = col * width/self.num_grids
    #     #     cv2.line(image, (int(c), 0), (int(c), height), (0, 0, 255), 1)
    #     # cv2.resize(image, (300, 300))
    #     tf.get_default_graph().finalize()
    #     return image

    def hot_mask(self, mask, raw_image):
        image = raw_image
        height, width = image.shape[0], image.shape[1]
        cell_height = height / self.num_grids
        cell_width = width / self.num_grids
        mask = (mask + 1) / 2.0
        # np.set_printoptions(edgeitems=1000000)
        for i in range(self.num_grids):
            for j in range(self.num_grids):
                image[int(i*cell_height):int((i+1)*cell_height), int(j*cell_width):int((j+1)*cell_width), :] = \
                    255. * mask[0, j, i, 0]
        return image

    def leaky_relu(self, alpha):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
        return op


# Model().train()
# Model(is_training=False).evaluate()
Model(is_training=False).evaluate_loop()
# Model(is_training=False).webcam()
# sess = tf.InteractiveSession()
# print(sess.run(Model().anchor_boxes()))
