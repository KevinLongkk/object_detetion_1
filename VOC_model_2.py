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
            'boat': 1.5,
            'bottle': 1.65,
            'bus': 1.2,
            'car': 1.0,
            'cat': 1.0,
            'chair': 1.2,
            'cow': 1.5,
            'diningtable': 0.55,  # 2.0
            'dog': 1.35,
            'horse': 1.3,
            'motorbike': 1.2,
            'person': 1.0,
            'pottedplant': 1.8,
            'sheep': 1.5,
            'sofa': 1.3,
            'train': 1.3,
            'tvmonitor': 1.4,
            'Vehicle': 1.0,
            'Animal': 1.2,
            'Indoor': 0.8,
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
        self.batch_size = 96
        self.batch_size_list = [16, 32, 64, 96, 108]
        # self.image_path = '/home/kevin/DataSet/VOCdevkit/VOC2007'
        # self.image_path = '/home/kevin/DataSet/bread'
        # self.image_path = '/home/kevin/DataSet/COCO/VOC_COCO_with_cls'
        # self.image_path = '/__DataSet/VOC/VOC2007'
        self.image_path = '/__DataSet/COCO'
        self.vgg_npz_path = './model/weight/vgg16_weights.npz'
        self.tensorboard_path = './test'

        self.num_grids = 17
        self.learning_rate = 1e-4
        self.save_path = './dense_log_2/'

        self.predict_per_cell = 2

        self.num_base_class = 4

        self.COORD_SCALE = 10.0
        self.OBJECT_SCALE = 8.0
        self.NOOBJECT_SCALE = 1.2
        self.CLASS_SCALE = 3.0

        self.num_anchors = 9
        self.num_class = 20
        self.is_training = is_training
        self.keep_prob = 1.0

        self.exclude_node = []

    def VGG_net(self, inputs):
        vgg = VGG_16.vgg16(inputs, self.predict_per_cell, weights='./model/weight/vgg16_weights.npz')
        net = vgg.convlayers()
        return net, vgg

    def networt(self, inputs):
        inputs = inputs * 2. - 1.0
        with slim.arg_scope([slim.conv2d], activation_fn=self.leaky_relu(0.1),
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=tf.constant_initializer(0.1)):
            # inputs += tf.random_normal(tf.shape(inputs), 0.001, 0.0003)
            net = slim.conv2d(inputs, 8, 5, padding='SAME', scope='conv1')

            net1_1 = slim.conv2d(net, 8, 17, 2, padding='SAME', scope='conv2_1')
            net1_2 = slim.conv2d(net, 8, 5, 2, padding='SAME', scope='conv2_2')
            net1_3 = slim.conv2d(net, 8, 3, 2, padding='SAME', scope='conv2_3')
            net1 = net1_1 + net1_2 + net1_3
            net2 = slim.max_pool2d(net, 2, padding='SAME', scope='pool1')
            net = tf.concat((net1, net2), axis=-1)

            shortcut = net
            net = slim.conv2d(net, 16, 3, padding='SAME', scope='conv3')
            net = slim.conv2d(net, 16, 3, padding='SAME', scope='conv4')
            net += shortcut
            # net = slim.max_pool2d(net, 2, 2, padding='SAME', scope='pool2')
            # net = slim.conv2d(net, 16, 3, padding='SAME', scope='conv5')
            # net = slim.conv2d(net, 16, 3, padding='SAME', scope='conv6')

            net1_1 = slim.conv2d(net, 16, 9, 2, padding='SAME', scope='conv6_2')
            # net1_2 = slim.conv2d(net, 16, 5, 2, padding='SAME', scope='conv6_3')
            # net1_3 = slim.conv2d(net, 16, 3, 2, padding='SAME', scope='conv6_4')
            net1 = net1_1
            net2 = slim.max_pool2d(net, 2, 2, padding='SAME', scope='pool3')
            net = tf.concat((net1, net2), axis=-1)

            shortcut = net
            net = slim.conv2d(net, 32, 3, padding='SAME', scope='conv7')
            net = slim.conv2d(net, 32, 3, padding='SAME', scope='conv8')
            net += shortcut

            net1_1 = slim.conv2d(net, 32, 7, 2, padding='SAME', scope='conv9_1')
            net1_2 = slim.conv2d(net, 32, 5, 2, padding='SAME', scope='conv9_2')
            net1_3 = slim.conv2d(net, 32, 3, 2, padding='SAME', scope='conv9_3')
            net1 = net1_1 + net1_2 + net1_3
            net2 = slim.max_pool2d(net, 2, 2, padding='SAME', scope='pool4')
            net = tf.concat((net1, net2), axis=-1)

            shortcut = net
            net = slim.conv2d(net, 64, 3, padding='SAME', scope='conv9_4')
            net = slim.conv2d(net, 64, 3, padding='SAME', scope='conv10')
            net += shortcut
            net = slim.conv2d(net, 128, 3, padding='SAME', scope='conv11')
            shortcut = net
            net = slim.conv2d(net, 128, 3, padding='SAME', scope='conv12')
            net = slim.conv2d(net, 128, 3, padding='SAME', scope='conv13')
            net += shortcut
            net1_1 = slim.conv2d(net, 128, 5, 2, padding='SAME', scope='conv14_1')
            # net1_2 = slim.conv2d(net, 128, 5, 2, padding='SAME', scope='conv14_2')
            # net1_3 = slim.conv2d(net, 128, 3, 2, padding='SAME', scope='conv14_3')
            net1 = net1_1
            net2 = slim.max_pool2d(net, 2, 2, padding='SAME', scope='pool5')
            net = tf.concat((net1, net2), axis=-1)

            shortcut = net
            net = slim.conv2d(net, 256, 3, padding='SAME', scope='conv15')
            net = slim.conv2d(net, 256, 3, padding='SAME', scope='conv16')
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
            net = slim.conv2d(net, 512, 3, padding='SAME', scope='conv24')
            net = slim.conv2d(net, 512, 1, padding='SAME', scope='conv25')
            net += shortcut
            net = slim.conv2d(net, 256, 3, padding='Valid', scope='conv26')
            # shortcut = net
            # net = slim.conv2d(net, 256, 3, padding='SAME', scope='conv27')
            # net = slim.conv2d(net, 256, 3, padding='SAME', scope='conv28')
            # net += shortcut
            net = slim.conv2d(net, 128, 3, padding='SAME', scope='conv29')
            shortcut = net
            net = slim.conv2d(net, 128, 1, padding='SAME', scope='conv30')
            net = slim.conv2d(net, 128, 3, padding='SAME', scope='conv31')
            net += shortcut
            # net = slim.conv2d(net, 64, 1, padding='SAME', scope='conv32')
            # shortcut = net
            # net = slim.conv2d(net, 64, 3, padding='SAME', scope='conv33')
            # net = slim.conv2d(net, 64, 1, padding='SAME', scope='conv34')
            # net += shortcut

            cls_net = slim.conv2d(net, (20 + 4) * self.predict_per_cell, 2, padding='SAME', scope='cls_conv1')
            cls_net = slim.conv2d(cls_net, (20 + 4) * self.predict_per_cell, 2, padding='SAME', scope='cls_conv2')
            cls_net = slim.conv2d(cls_net, (20 + 4) * self.predict_per_cell, 3, padding='SAME', scope='cls_conv3')
            cls_net = slim.conv2d(cls_net, (20 + 4) * self.predict_per_cell, 1, padding='SAME', scope='cls_conv4')

            box_net = slim.conv2d(net, 45 * self.predict_per_cell, 2, padding='SAME', scope='box_conv1')
            box_net = slim.conv2d(box_net, 45 * self.predict_per_cell, 1, padding='SAME', scope='box_conv2')

            net = tf.concat((box_net, cls_net), axis=-1)
            return net

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

    def new_loss(self, labels, predictions):
        # predictions = tf.maximum(predictions, 0.0)
        anchor_boxes = self.anchor_boxes()
        raw_labels = labels
        raw_predictions = tf.reshape(predictions, [self.batch_size, self.num_grids, self.num_grids,
                                                   self.predict_per_cell, 65+self.num_base_class])
        labels = tf.tile(labels[..., :5], [1, 1, 1, 1, self.num_anchors])
        labels = tf.reshape(labels, (self.batch_size, self.num_grids, self.num_grids, self.predict_per_cell,
                                     self.num_anchors, 5))

        predictions_ = tf.reshape(raw_predictions[..., :45], (self.batch_size, self.num_grids, self.num_grids,
                                                              self.predict_per_cell, self.num_anchors, 5))
        predictions_ab = tf.stack([
            predictions_[:, :, :, :, 0] * [1., 1., 1., tf.cast(anchor_boxes[0][0], dtype=tf.float32),
                                           tf.cast(anchor_boxes[0][1], dtype=tf.float32)],
            predictions_[:, :, :, :, 1] * [1., 1., 1., tf.cast(anchor_boxes[1][0], dtype=tf.float32),
                                           tf.cast(anchor_boxes[1][1], dtype=tf.float32)],
            predictions_[:, :, :, :, 2] * [1., 1., 1., tf.cast(anchor_boxes[2][0], dtype=tf.float32),
                                           tf.cast(anchor_boxes[2][1], dtype=tf.float32)],
            predictions_[:, :, :, :, 3] * [1., 1., 1., tf.cast(anchor_boxes[3][0], dtype=tf.float32),
                                           tf.cast(anchor_boxes[3][1], dtype=tf.float32)],
            predictions_[:, :, :, :, 4] * [1., 1., 1., tf.cast(anchor_boxes[4][0], dtype=tf.float32),
                                           tf.cast(anchor_boxes[4][1], dtype=tf.float32)],
            predictions_[:, :, :, :, 5] * [1., 1., 1., tf.cast(anchor_boxes[5][0], dtype=tf.float32),
                                           tf.cast(anchor_boxes[5][1], dtype=tf.float32)],
            predictions_[:, :, :, :, 6] * [1., 1., 1., tf.cast(anchor_boxes[6][0], dtype=tf.float32),
                                           tf.cast(anchor_boxes[6][1], dtype=tf.float32)],
            predictions_[:, :, :, :, 7] * [1., 1., 1., tf.cast(anchor_boxes[7][0], dtype=tf.float32),
                                           tf.cast(anchor_boxes[7][1], dtype=tf.float32)],
            predictions_[:, :, :, :, 8] * [1., 1., 1., tf.cast(anchor_boxes[8][0], dtype=tf.float32),
                                           tf.cast(anchor_boxes[8][1], dtype=tf.float32)]
        ], axis=4)

        iou_predict_truth = self.cal_iou(predictions_ab[..., 1:], labels[..., 1:])

        truth_mask = tf.tile(tf.expand_dims(raw_labels[..., 0], 4), [1, 1, 1, 1, self.num_anchors])

        iou_mask = tf.reduce_max(iou_predict_truth, axis=4, keep_dims=True)
        mask = tf.cast(iou_predict_truth >= tf.minimum(0.9, iou_mask), dtype=tf.float32) * truth_mask

        noobj_mask = 1 - mask
        threshold_noobj_mask = tf.cast(predictions_ab[..., 0] >= 0.2, dtype=tf.float32) * noobj_mask

        coordinate_loss = tf.reduce_mean(tf.reduce_sum((tf.square(labels[..., 1:] -
                                                                  predictions_ab[..., 1:]) * tf.expand_dims(mask, 5)),
                                                       axis=[1, 2, 3, 4, 5])) * self.COORD_SCALE

        noobj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(predictions_ab[..., 0]) * threshold_noobj_mask,
                                                  axis=[1, 2, 3, 4])) * self.NOOBJECT_SCALE

        negative_mask = tf.cast(tf.logical_and(predictions_ab < 0., predictions_ab > -1.0),
                                dtype=tf.float32)
        noobj_loss_ = tf.reduce_mean(tf.reduce_sum(-tf.log(1 + predictions_ab * negative_mask),
                                                   axis=[1, 2, 3, 4, 5]))

        obj_loss = tf.reduce_mean(tf.reduce_sum(-tf.log(tf.clip_by_value(predictions_ab[..., 0], 1e-10, 1.0)) *
                                                tf.square(1 - predictions_ab[..., 0]) * mask,
                                                axis=[1, 2, 3, 4])) * self.OBJECT_SCALE

        # cls_loss = tf.reduce_mean(tf.reduce_sum(
        #     tf.nn.softmax_cross_entropy_with_logits(labels=raw_labels[..., 5:25],
        #                                             logits=raw_predictions[..., 45:65]) * raw_labels[..., 0],
        #                                             axis=[1, 2, 3])) * self.CLASS_SCALE
        base_cls_loss = tf.reduce_mean(tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=raw_labels[..., 25:],
                                                    logits=raw_predictions[..., 65:]) * raw_labels[..., 0],
                                                    axis=[1, 2, 3])) * self.CLASS_SCALE * 2.
        cls_loss = tf.reduce_mean(tf.reduce_sum(
            tf.square(raw_labels[..., 5:25] - raw_predictions[..., 45:65]) * tf.expand_dims(raw_labels[..., 0], 4),
            axis=[1, 2, 3, 4])) * self.CLASS_SCALE
        # base_cls_loss = tf.reduce_mean(tf.reduce_sum(
        #     tf.square(raw_labels[..., 25:] - raw_predictions[..., 65:]) * tf.expand_dims(raw_labels[..., 0], 4),
        #     axis=[1, 2, 3, 4])) * self.CLASS_SCALE * 3

        # obj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(iou_predict_truth-predictions_ab[..., 0]) * mask,
        #                                         axis=[1, 2, 3])) * self.OBJECT_SCALE

        # losses = tf.reduce_mean(tf.reduce_sum(tf.square(predictions_ab-labels), axis=[1, 2, 3, 4]))
        # obj_loss_ = tf.reduce_mean(tf.reduce_sum(tf.square(mask-predictions_ab[..., 0]) * mask,
        #                                          axis=[1, 2, 3])) * 100.

        # losses = noobj_loss_ + cls_loss + base_cls_loss

        losses = coordinate_loss + noobj_loss + obj_loss + noobj_loss_ + cls_loss + base_cls_loss
        # losses = tf.reduce_mean(tf.reduce_sum(tf.square(labels - predictions_), axis=[1, 2, 3, 4, 5])) + cls_loss

        tf.summary.scalar('coordinate_loss', coordinate_loss)
        tf.summary.scalar('noobj_loss', noobj_loss)
        tf.summary.scalar('obj_loss', obj_loss)
        tf.summary.scalar('cls_loss', cls_loss)
        tf.summary.scalar('base_cls_loss', base_cls_loss)
        tf.summary.scalar('total_loss', losses)
        # tf.summary.scalar('noobj_loss_', noobj_loss_)
        # tf.summary.scalar('obj_loss_', obj_loss_)

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

        tf.summary.image('image', inputs_ph, 10)

        train_data = data.Data(self.image_path, self.batch_size, self.image_size)

        # predictions, vgg = self.VGG_net(inputs_ph)
        predictions = self.networt(inputs_ph)
        loss = self.new_loss(labels_ph, predictions)

        global_step = tf.Variable(0, trainable=False)
        global_epoch = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 5000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        # grads = optimizer.compute_gradients(loss)
        # for i, (g, v) in enumerate(grads):
        #     if g is not None:
        #         grads[i] = (tf.clip_by_norm(g, 5), v)  # 阈值这里设为5
        # train_op = optimizer.apply_gradients(grads, global_step)

        s_saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

        with tf.Session() as sess:
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
            for i in range(1, 30000):
                time_ = time.time()
                # batch_size = self.batch_size_list[int((random.random()-1e-10) * 5)]
                images, labels = train_data.load_data(0)
                # images, labels = train_data.load_test_data(0)
                labels = self.labels_handler(labels)
                read_data_time = time.time() - time_
                # labels = np.ones_like(labels)
                # np.set_printoptions(edgeitems=1000000)
                # print(labels[0, 4, 4, :, :])
                # break

                _, step, epoch, losses = sess.run([optimizer, global_step, global_epoch, loss],
                                           feed_dict={inputs_ph: images, labels_ph: labels})
                sum += losses
                totall_time = time.time() - time_
                if i % 10 == 0:
                    llosses = sum / 10.
                    sum = 0
                    print('Global step: %d, 10 steps mean loss is: %f, one step read_data_time: %f, one step totall_time: %f'
                          % (step, llosses, read_data_time, totall_time))
                    # global_epoch = global_epoch.assign(train_data.epoch+epoch)
                # if i % 40 == 0:
                    summary_str = sess.run(merged_summary_op, feed_dict={inputs_ph: images, labels_ph: labels})
                    summary_writer.add_summary(summary_str, global_step=step)
                if i % 500 == 0:
                    s_saver.save(sess, self.save_path, global_step=global_step)
                    print('save model success')

    def evaluate(self):

        raw_image = cv2.imread('./image/001981.jpg')
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
            raw_result = sess.run(predictions, feed_dict={input: [image]})
            raw_result = np.reshape(raw_result, [1, self.num_grids, self.num_grids,
                                                 self.predict_per_cell, 65+self.num_base_class])
            result = np.reshape(raw_result[..., :45], (1, self.num_grids, self.num_grids,
                                                       self.predict_per_cell, self.num_anchors, 5))
            result_cls = raw_result[..., 45:]
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
        image_path = '/home/kevin/DataSet/VOCdevkit/VOC_test/VOC2010_test/VOC2010/JPEGImages'
        # image_path = '/home/kevin/DataSet/COCO/VOC_COCO_with_cls/2017/JPEGImages'
        # image_path = '/home/kevin/DataSet/bread/JPEGImages'
        # image_path = '/home/kevin/DataSet/VOCdevkit/VOC2007/JPEGImages'
        image_list = os.listdir(image_path)
        random.shuffle(image_list)
        # image_list.sort()
        input = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3))

        # predictions, vgg = self.VGG_net(input)
        predictions = self.networt(input)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('evaluate...')
            print('restoring from', tf.train.latest_checkpoint(self.save_path))
            saver.restore(sess, tf.train.latest_checkpoint(self.save_path))
            for img_path in image_list:
                raw_image = cv2.imread(os.path.join(image_path, img_path))
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.image_size, self.image_size)) / 255.0
                raw_result = sess.run(predictions, feed_dict={input: [image]})
                raw_result = np.reshape(raw_result, [1, self.num_grids, self.num_grids,
                                                     self.predict_per_cell, 65+self.num_base_class])
                result = np.reshape(raw_result[..., :45], (1, self.num_grids, self.num_grids,
                                                           self.predict_per_cell, self.num_anchors, 5))
                result_cls = raw_result[..., 45:]
                for i in range(self.num_grids):
                    for j in range(self.num_grids):
                        for m in range(self.predict_per_cell):
                            k = np.where(result[0, i, j, m, :, 0] == np.max(result[0, i, j, m, :, 0]))
                            if len(k[0]) > 1:
                                print(str(i * self.num_grids + j) + '_%d' % m, 0)
                                continue
                            # k = result[0, i, j, :, 0].index(max(result[0, i, j, :, 0]))
                            # if result[0, i, j, 0] > -0.01:
                            if True:
                                # print(result.shape)
                                print(str(i * self.num_grids + j) + '_%d' % m, str(result[0, i, j, m, k, :]))
                                # print(str(result[0, i, j, :]))
                image = self.visual(raw_image, result[0], result_cls[0], threshold=0.25)
                cv2.imshow(' ', image)
                if cv2.waitKey(0) == 27:
                    break

    def webcam(self):
        # cameraCapture = cv2.VideoCapture('./image/03.avi')
        cameraCapture = cv2.VideoCapture(0)

        input = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3))
        predictions = self.networt(input)
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
                raw_result = sess.run(predictions, feed_dict={input: [frame]})
                raw_result = np.reshape(raw_result, [1, self.num_grids, self.num_grids, self.predict_per_cell, 65+4])
                result = np.reshape(raw_result[..., :45], (1, self.num_grids, self.num_grids,
                                                           self.predict_per_cell, self.num_anchors, 5))
                result_cls = raw_result[..., 45:]
                image = self.visual(raw_frame, result[0], result_cls[0], threshold=0.3)
                cv2.imshow(' ', image)
                if cv2.waitKey(20) == 27:
                    break

    def visual(self, image, labels, result_cls, threshold=0.5):
        tf.reset_default_graph()
        with tf.Session() as sess:
            anchor_boxes = sess.run(self.anchor_boxes())
            boxes, scores, cls = [], [], []
            width, height = image.shape[1], image.shape[0]
            for i in range(self.num_grids):
                for j in range(self.num_grids):
                    for m in range(self.predict_per_cell):
                        k = np.where(labels[i, j, m, :, 0] == np.max(labels[i, j, m, :, 0]))
                        k = k[0][0]
                        # if i*19+j==104:
                        if labels[i, j, m, k, 0] > threshold:
                            print(str(i * self.num_grids + j) + '_' + str(m))
                            center_x = (labels[i, j, m, k, 1] + i) / self.num_grids
                            center_y = (labels[i, j, m, k, 1] + j) / self.num_grids
                            xmin, xmax = int((center_x - labels[i, j, m, k, 3] / 2 * anchor_boxes[k][0]) * width), \
                                         int((center_x + labels[i, j, m, k, 3] / 2 * anchor_boxes[k][0]) * width)
                            ymin, ymax = int((center_y - labels[i, j, m, k, 4] / 2 * anchor_boxes[k][1]) * height), \
                                         int((center_y + labels[i, j, m, k, 4] / 2 * anchor_boxes[k][1]) * height)
                            xmin = 1 if xmin <= 0 else xmin
                            ymin = 1 if ymin <= 0 else ymin
                            xmax = width - 1 if xmax >= width else xmax
                            ymax = height - 1 if ymax >= height else ymax
                            coord = [ymin, xmin, ymax, xmax]
                            boxes.append(coord)
                            scores.append(labels[i, j, m, k, 0])

                            print(result_cls[i, j, m, 20:])
                            print(result_cls[i, j, m, :20])
                            base_class = sess.run(tf.arg_max(result_cls[i, j, m, 20:], -1))
                            base_cls_list, cls_list, cls_key = None, [], None
                            for key in self.base_class.keys():
                                if self.base_class[key][0] == base_class:
                                    base_cls_list = list(self.base_class[key][1])
                                    cls_key = key
                                    break
                            for c in base_cls_list:
                                cls_list.append(result_cls[i, j, m, c])
                            cls.append(self.base_class[cls_key][1][sess.run(tf.arg_max(cls_list, -1))])
                            # cls.append(sess.run(tf.arg_max(cls_list, -1)))
            try:
                truth_boxes = tf.image.non_max_suppression(np.array(boxes), np.array(scores), 10, 0.5)
                truth_boxes = sess.run(truth_boxes)
                # cls = sess.run(cls)
                for i in truth_boxes:
                    # r, g, b = random.random(), random.random(), random.random()
                    cv2.rectangle(image, (boxes[i][1], boxes[i][0]), (boxes[i][3], boxes[i][2]), (0, 255, 0), 1)
                    for k in self.VOC_LABELS.keys():
                        if self.VOC_LABELS[k] == cls[i]:
                            print(str(k)+str(scores[i])[:4])
                            cv2.putText(image, str(k)+str(scores[i])[:4], (boxes[i][1], boxes[i][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                            # cv2.putText(image, str(k), (boxes[i][1], boxes[i][0]),
                            #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            except:
                print('No bbox')
        tf.get_default_graph().finalize()
        return image

    def leaky_relu(self, alpha):
        def op(inputs):
            return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
        return op


Model().train()
# Model().evaluate()
# Model().evaluate_loop()
# Model().webcam()
# sess = tf.InteractiveSession()
# print(sess.run(Model().anchor_boxes()))