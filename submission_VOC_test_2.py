import os
import tensorflow as tf
import VOC_model_2 as VOC_model
import cv2
import numpy as np
import time

class evaluate(object):

    def __init__(self, test_file, test_flag, save_path):
        self.test_file = test_file
        self.test_flag = test_flag
        self.check(self.test_flag)
        self.save_path = save_path

        self.checkpoint_path = '/media/kevin/0E5B14B00E5B14B0/VOC_fine_tune/'

        self.test_format = 'comp3_det_test_'
        self.val_format = 'comp3_det_val_'

        self.VOC_model = VOC_model.Model(is_training=False)
        self.anchor_boxes = self.VOC_model.anchor_boxes
        self.predict_per_cell = self.VOC_model.predict_per_cell
        self.image_size = self.VOC_model.image_size
        self.num_grids = self.VOC_model.num_grids
        self.num_anchors = self.VOC_model.num_anchors
        self.base_class = self.VOC_model.base_class

        self.inputs = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])

        self.network = self.VOC_model.networt(self.inputs)

        self.threshold = 0.40

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

        self.result_str = [''] * 20


    def check(self, test_flag):
        assert test_flag in ['VOC2008', 'VOC2009', 'VOC2010', 'VOC2011', 'VOC2012', 'VOC2008_validation',
                                  'VOC2009_validation', 'VOC2010_validation', 'VOC2012_validation']
        self.test_or_val_flag = self.test_flag[:7]

    def make_files(self, format):

        # format = test_format
        os.makedirs(os.path.join(self.save_path, 'results', self.test_or_val_flag, 'Main'))
        for label in self.VOC_LABELS.keys():
            file = open(os.path.join(self.save_path, 'results', self.test_or_val_flag, 'Main', format+label+'.txt'), 'w')
            file.close()
        print('make files success')

    def evaluate_loop(self, format):

        # image = cv2.imread('./image/000030.jpg')
        image_path = os.path.join(self.test_file, 'JPEGImages')
        if self.test_flag.find('validation') != -1:
            txt_name = 'aeroplane_val.txt'
        else:
            txt_name = 'aeroplane_test.txt'
        with open(os.path.join(self.test_file, 'ImageSets', 'Main', txt_name)) as txt:
            lines = txt.readlines()
            image_list = []
            for line in lines:
                image_list.append(line.split(' ')[0]+'.jpg')
                print(line.split(' ')[0]+'.jpg')

        # image_list = os.listdir(image_path)
        image_list.sort()

        predictions, _ = self.network
        raw_predictions = tf.concat((
            tf.reshape(predictions[..., :48], [1, self.num_grids, self.num_grids,
                                               self.predict_per_cell, 24]),
            tf.reshape(predictions[..., 48:], [1, self.num_grids, self.num_grids,
                                               self.predict_per_cell, 45])), axis=-1)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('evaluate...')
            print('restoring from', tf.train.latest_checkpoint(self.checkpoint_path))
            saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_path))
            index = 0

            anchor_boxes = sess.run(self.anchor_boxes())
            for img_path in image_list:
                # print(img_path)
                raw_image = cv2.imread(os.path.join(image_path, img_path))
                # print(os.path.join(image_path, img_path))
                width, height = raw_image.shape[1], raw_image.shape[0]
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.image_size, self.image_size)) / 255.0
                time_ = time.time()

                raw_result = sess.run(raw_predictions, feed_dict={self.inputs: [image]})
                # print(image)
                # print(raw_result)
                # raw_result = np.reshape(raw_result, [1, self.num_grids, self.num_grids, self.predict_per_cell, 65+4])
                sess_time = time.time() - time_

                result = np.reshape(raw_result[..., 24:], (1, self.num_grids, self.num_grids,
                                                           self.predict_per_cell, self.num_anchors, 5))
                result_cls = raw_result[..., :24]
                # result = np.reshape(raw_result[..., :45], (1, self.num_grids, self.num_grids, self.predict_per_cell,
                #                                            self.num_anchors, 5))
                # result_cls = raw_result[..., 45:]
                self.visual(anchor_boxes, img_path, result[0], result_cls[0], width, height,
                            threshold=self.threshold, image=raw_image)
                print(index, 'sess time:', sess_time)
                if index % 200 == 0:
                    files = self.open_files(format)
                    for i in range(20):
                        print(self.result_str[i])
                        files[i].write(self.result_str[i])
                        files[i].close()
                        self.result_str[i] = ''
                index += 1
            files = self.open_files(format)
            for i in range(20):
                files[i].write(self.result_str[i])
                files[i].close()

    def visual(self, anchor_boxes, img_path, labels, result_cls, width, height, threshold=0.5, image=None):
        tf.reset_default_graph()
        with tf.Session() as sess:
            boxes, scores, cls = [], [], []
            for i in range(self.num_grids):
                for j in range(self.num_grids):
                    for m in range(self.predict_per_cell):
                        k = np.where(labels[i, j, m, :, 0] == np.max(labels[i, j, m, :, 0]))
                        k = k[0][0]
                        if labels[i, j, m, k, 0] > threshold:
                            labels[i, j, m, k, 1:3] = np.square(labels[i, j, m, k, 1:3])
                            center_x = (labels[i, j, m, k, 1] + i) / self.num_grids
                            center_y = (labels[i, j, m, k, 2] + j) / self.num_grids
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
                            # cls.append(tf.arg_max(result_cls[i, j, m], -1))
                            base_class = sess.run(tf.arg_max(result_cls[i, j, m, :4], -1))
                            base_cls_list, cls_list, cls_key = None, [], None
                            for key in self.base_class.keys():
                                if self.base_class[key][0] == base_class:
                                    base_cls_list = list(self.base_class[key][1])
                                    cls_key = key
                                    break
                            for c in base_cls_list:
                                cls_list.append(result_cls[i, j, m, c+4])
                            cls.append(self.base_class[cls_key][1][sess.run(tf.arg_max(cls_list, -1))])

            try:
                truth_boxes = tf.image.non_max_suppression(np.array(boxes), np.array(scores), 10, 0.5)
                # cls = sess.run(cls)
                truth_boxes = sess.run(truth_boxes)
                for i in truth_boxes:
                    # r, g, b = random.random(), random.random(), random.random()
                    xmin, ymin = boxes[i][1], boxes[i][0]
                    xmax, ymax = boxes[i][3], boxes[i][2]
                    # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    # cv2.imshow(' ', image)
                    # print(scores[i], xmin, ymin, xmax, ymax)
                    # cv2.waitKey(0)
                    for k in self.VOC_LABELS.keys():
                        if self.VOC_LABELS[k] == cls[i]:
                            self.result_str[self.VOC_LABELS[k]] += \
                                '%s %f %f %f %f %f\n' % (img_path[:-4], scores[i], xmin, ymin, xmax, ymax)
                            break
            except:
                print('No bbox')
            tf.get_default_graph().finalize()

    def open_files(self, format):
        files = []
        for k in sorted(self.VOC_LABELS.keys()):
            file = open(os.path.join(self.save_path, 'results', self.test_or_val_flag, 'Main', format + k + '.txt'), 'a+')
            files.append(file)
        return files

    def main(self):
        if self.test_flag.find('validation') != -1:
            format = self.val_format
        else:
            format = self.test_format
        self.make_files(format)
        self.evaluate_loop(format)


evaluate(test_file='/home/kevin/DataSet/VOCdevkit/VOC_test/VOC2012_test/VOC2012',
         test_flag='VOC2012', save_path='/home/kevin/DataSet/VOCdevkit/VOC_test/submission/SAVE/2019_01_04_VOC2012').main()

