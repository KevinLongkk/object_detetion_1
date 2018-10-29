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

        self.checkpoint_path = './log'

        self.test_format = 'comp3_det_test_'
        self.val_format = 'comp3_det_val_'

        self.image_size = 300
        self.num_grids = 10
        self.num_anchors = 9
        self.inputs = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3])

        self.VOC_model = VOC_model.Model()
        self.network, _ = self.VOC_model.VGG_net(self.inputs)

        self.anchor_boxes = self.VOC_model.anchor_boxes
        self.predict_per_cell = self.VOC_model.predict_per_cell

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

        predictions = self.network

        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('evaluate...')
            print('restoring from', tf.train.latest_checkpoint(self.checkpoint_path))
            saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_path))
            index = 0

            anchor_boxes = sess.run(self.anchor_boxes())
            for img_path in image_list:
                image = cv2.imread(os.path.join(image_path, img_path))
                # print(os.path.join(image_path, img_path))
                width, height = image.shape[1], image.shape[0]
                image = cv2.resize(image, (self.image_size, self.image_size)) / 255.0
                time_ = time.time()
                raw_result = sess.run(predictions, feed_dict={self.inputs: [image]})
                raw_result = np.reshape(raw_result, [1, self.num_grids, self.num_grids, self.predict_per_cell, 65])
                sess_time = time.time() - time_
                result = np.reshape(raw_result[..., :45], (1, self.num_grids, self.num_grids, self.predict_per_cell,
                                                           self.num_anchors, 5))
                result_cls = raw_result[..., 45:]
                self.visual(anchor_boxes, img_path, result[0], result_cls[0], width, height, threshold=0.3)
                print(index, 'sess time:', sess_time, )
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

    def visual(self, anchor_boxes, img_path, labels, result_cls, width, height, threshold=0.5):
        tf.reset_default_graph()
        with tf.Session() as sess:
            boxes, scores, cls = [], [], []
            for i in range(self.num_grids):
                for j in range(self.num_grids):
                    for m in range(self.predict_per_cell):
                        k = np.where(labels[i, j, m, :, 0] == np.max(labels[i, j, m, :, 0]))
                        k = k[0][0]
                        if labels[i, j, m, k, 0] > threshold:
                            center_x = (labels[i, j, m, k, 1] + i) / self.num_grids
                            center_y = (labels[i, j, m, k, 1] + j) / self.num_grids
                            xmin, xmax = int((center_x - labels[i, j, m, k, 3] / 2 * anchor_boxes[k][0]) * self.image_size), \
                                         int((center_x + labels[i, j, m, k, 3] / 2 * anchor_boxes[k][0]) * self.image_size)
                            ymin, ymax = int((center_y - labels[i, j, m, k, 4] / 2 * anchor_boxes[k][1]) * self.image_size), \
                                         int((center_y + labels[i, j, m, k, 4] / 2 * anchor_boxes[k][1]) * self.image_size)
                            coord = [ymin, xmin, ymax, xmax]
                            boxes.append(coord)
                            scores.append(labels[i, j, m, k, 0])
                            cls.append(tf.arg_max(result_cls[i, j, m], -1))

            try:
                truth_boxes = tf.image.non_max_suppression(np.array(boxes), np.array(scores), 10, 0.3)
                cls = sess.run(cls)
                truth_boxes = sess.run(truth_boxes)
                for i in truth_boxes:
                    # r, g, b = random.random(), random.random(), random.random()
                    xmin, ymin = max(boxes[i][1]*width/self.image_size, 0.), max(boxes[i][0]*height/self.image_size, 0.)
                    xmax, ymax = min(boxes[i][3]*width/self.image_size, float(width)), \
                                 min(boxes[i][2]*height/self.image_size, float(height))
                    for k in self.VOC_LABELS.keys():
                        if self.VOC_LABELS[k] == cls[i]:
                            self.result_str[self.VOC_LABELS[k]] += \
                                '%s %f %f %f %f %f\n' % (img_path[:-4], scores[i], xmin+1., ymin+1., xmax, ymax)
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


evaluate(test_file='/home/kevin/DataSet/VOCdevkit/VOC2012',
         test_flag='VOC2012_validation', save_path='/home/kevin/DataSet/VOCdevkit/VOC_test/submission/SAVE/2018_10_21_VOC2012_val').main()

