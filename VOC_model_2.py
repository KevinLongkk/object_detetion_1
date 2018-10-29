import tensorflow as tf
from OpenDL.Data import get_pascal_voc_data as data
from tensorflow.contrib import slim
import cv2
import numpy as np
import os
import random
from model import VGG_16


class Model(object):

    def __init__(self, is_training=True):

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

        self.image_size = 448
        self.batch_size = 64
        # self.image_path = '/home/kevin/DataSet/VOCdevkit/VOC2007'
        # self.image_path = '/home/kevin/DataSet/bread'
        self.image_path = '/home/kevin/DataSet/COCO/VOC_COCO_with_cls/2017'
        self.vgg_npz_path = './model/weight/vgg16_weights.npz'
        self.tensorboard_path = './dense_tensorboard'

        self.num_grids = 13
        self.learning_rate = 1e-5
        self.save_path = './dense_log/'

        self.predict_per_cell = 2

        self.COORD_SCALE = 13.0
        self.OBJECT_SCALE = 6.0
        self.NOOBJECT_SCALE = 4.0
        self.CLASS_SCALE = 5.0

        self.num_anchors = 9
        self.num_class = 20
        self.is_training = is_training
        self.keep_prob = 1.0

        self.exclude_node = ['add_conv6_2', 'add_conv6_3', 'add_conv6_4', 'add_conv_7', 'conv7_1', 'conv7_2', 'conv7_3', 'conv7_4']

    def VGG_net(self, inputs):
        vgg = VGG_16.vgg16(inputs, self.predict_per_cell, weights='./model/weight/vgg16_weights.npz')
        net = vgg.convlayers()
        return net, vgg

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
                                                   self.predict_per_cell, 65])
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

        # iou_mask = iou_predict_truth * tf.tile(tf.expand_dims(raw_labels[..., 0], 3), [1, 1, 1, self.num_anchors])

        truth_mask = tf.tile(tf.expand_dims(raw_labels[..., 0], 4), [1, 1, 1, 1, self.num_anchors])

        iou_mask = tf.reduce_max(iou_predict_truth, axis=4, keep_dims=True)
        mask = tf.cast(iou_predict_truth >= tf.minimum(0.8, iou_mask), dtype=tf.float32) * truth_mask

        # mask = tf.tile(tf.expand_dims(tf.reduce_max(iou_predict_truth, 3) * raw_labels[..., 0], 3),
        #                [1, 1, 1, self.num_anchors])
        noobj_mask = 1 - mask
        threshold_noobj_mask = tf.cast(predictions_ab[..., 0] >= 0.2, dtype=tf.float32) * noobj_mask
        # noobj_mask = 1 - tf.tile(tf.expand_dims(raw_labels[..., 0], 3), [1., 1., 1., self.num_anchors])

        coordinate_loss = tf.reduce_mean(tf.reduce_sum((tf.square(labels[..., 1:] -
                                                                  predictions_ab[..., 1:]) * tf.expand_dims(mask, 5)),
                                                       axis=[1, 2, 3, 4, 5])) * self.COORD_SCALE
        # anchor_loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels[..., 3:] -
        #                 predictions_ab[..., 3:]) * tf.expand_dims(iou_mask, 4), axis=[1, 2, 3, 4])) * self.ANCHOR_SCALE
        noobj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(predictions_ab[..., 0]) * threshold_noobj_mask,
                                                  axis=[1, 2, 3, 4])) * self.NOOBJECT_SCALE

        # noobj_loss = tf.reduce_mean(tf.reduce_sum(-tf.log(tf.clip_by_value(
        #     1-predictions_ab[..., 0], 1e-10, 1.0)) * noobj_mask, axis=[1, 2, 3]))
        negative_mask = tf.cast(tf.logical_and(predictions_ab < 0., predictions_ab > -1.0),
                                dtype=tf.float32)
        noobj_loss_ = tf.reduce_mean(tf.reduce_sum(-tf.log(1 + predictions_ab * negative_mask),
                                                   axis=[1, 2, 3, 4, 5]))

        obj_loss = tf.reduce_mean(tf.reduce_sum(-tf.log(tf.clip_by_value(predictions_ab[..., 0], 1e-10, 1.0)) *
                                                tf.square(1 - predictions_ab[..., 0]) * mask,
                                                axis=[1, 2, 3, 4])) * self.OBJECT_SCALE

        cls_loss = tf.reduce_mean(tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=raw_labels[..., 5:],
                                                    logits=raw_predictions[..., 45:]) * raw_labels[..., 0],
                                                    axis=[1, 2, 3])) * self.CLASS_SCALE
        # cls_loss = tf.reduce_mean(tf.reduce_sum(
        #     tf.square(raw_labels[..., 5:] - raw_predictions[..., 45:]) * tf.expand_dims(raw_labels[..., 0], 4),
        #     axis=[1, 2, 3, 4])) * self.CLASS_SCALE

        # obj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(iou_predict_truth-predictions_ab[..., 0]) * mask,
        #                                         axis=[1, 2, 3])) * self.OBJECT_SCALE

        # losses = tf.reduce_mean(tf.reduce_sum(tf.square(predictions_ab-labels), axis=[1, 2, 3, 4]))
        # obj_loss_ = tf.reduce_mean(tf.reduce_sum(tf.square(mask-predictions_ab[..., 0]) * mask,
        #                                          axis=[1, 2, 3])) * 100.

        # losses = obj_loss + noobj_loss + noobj_loss_ + cls_loss

        losses = coordinate_loss + noobj_loss + obj_loss + noobj_loss_ + cls_loss
        # losses = tf.reduce_mean(tf.reduce_sum(tf.square(labels - predictions_), axis=[1, 2, 3, 4, 5])) + cls_loss

        tf.summary.scalar('coordinate_loss', coordinate_loss)
        tf.summary.scalar('noobj_loss', noobj_loss)
        tf.summary.scalar('obj_loss', obj_loss)
        tf.summary.scalar('cls_loss', cls_loss)
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
        s_labels = np.zeros((self.batch_size, self.num_grids, self.num_grids, self.predict_per_cell, 25), dtype=np.float32)
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
                        s_labels[i, x_ind, y_ind, k, self.VOC_LABELS[label[0]] + 5] = 1.
                        break
        return s_labels

    def train(self):
        print('loading data from :' + self.image_path)
        inputs_ph = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3))
        labels_ph = tf.placeholder(tf.float32, (None, self.num_grids, self.num_grids, self.predict_per_cell, 25))

        tf.summary.image('image', inputs_ph, 10)

        train_data = data.Data(self.image_path, self.batch_size, self.image_size)

        predictions, vgg = self.VGG_net(inputs_ph)
        loss = self.new_loss(labels_ph, predictions)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 5000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

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
                if sess.run(global_step) == 0:
                    print('loading weight from %s' % self.vgg_npz_path)
                    vgg.load_weights(self.vgg_npz_path, sess)
            sum = 0
            for i in range(1, 30000):
                images, labels = train_data.load_data(data_augmentation=0)
                # images, labels = train_data.load_test_data(0)
                labels = self.labels_handler(labels)
                # labels = np.ones_like(labels)
                # np.set_printoptions(edgeitems=1000000)
                # print(labels[0, 4, 4, :, :])
                # break

                _, step, losses = sess.run([optimizer, global_step, loss],
                                           feed_dict={inputs_ph: images, labels_ph: labels})
                sum += losses
                if i % 10 == 0:
                    llosses = sum / 10.
                    sum = 0
                    print('Global step: %d, 10 steps mean loss is: %f' % (step, llosses))
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

        predictions, vgg = self.VGG_net(input)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('evaluate...')
            print('restoring from', tf.train.latest_checkpoint(self.save_path))
            saver.restore(sess, tf.train.latest_checkpoint(self.save_path))
            raw_result = sess.run(predictions, feed_dict={input: [image]})
            raw_result = np.reshape(raw_result, [1, self.num_grids, self.num_grids, self.predict_per_cell, 65])
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
        # image_path = '/home/kevin/DataSet/VOCdevkit/VOC_test/VOC2010_test/VOC2010/JPEGImages'
        # image_path = '/home/kevin/DataSet/COCO/VOC_COCO_with_cls/JPEGImages'
        # image_path = '/home/kevin/DataSet/bread/JPEGImages'
        image_path = '/home/kevin/DataSet/VOCdevkit/VOC2007/JPEGImages'
        image_list = os.listdir(image_path)
        image_list.sort()
        input = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3))

        predictions, vgg = self.VGG_net(input)

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
                raw_result = np.reshape(raw_result, [1, self.num_grids, self.num_grids, self.predict_per_cell, 65])
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
        # cameraCapture = cv2.VideoCapture('./image/01.avi')
        cameraCapture = cv2.VideoCapture(0)

        input = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3))
        predictions, _ = self.VGG_net(input)
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
                raw_result = np.reshape(raw_result, [1, self.num_grids, self.num_grids, self.predict_per_cell, 65])
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
                            cls.append(tf.arg_max(result_cls[i, j, m], -1))
            try:
                truth_boxes = tf.image.non_max_suppression(np.array(boxes), np.array(scores), 20, 0.5)
                truth_boxes = sess.run(truth_boxes)
                cls = sess.run(cls)
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