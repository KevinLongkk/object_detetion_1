from OpenDL.Data import get_pascal_voc_data as data
import VOC_model
import tensorflow as tf
import numpy as np
import cv2

class evaluation(object):

    def __init__(self, test_file):
        self.test_file = test_file
        self.batch_size = 1
        self.image_size = 300
        self.save_path = './log'
        self.threshold = 0.4
        self.num_grids = 19
        self.iou_threshold = 0.5

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

    def cal_AP(self, labels, predictions):
        for label, prediction in zip(labels, predictions):
            pre_boxes = []
            for i in range(self.num_grids):
                for j in range(self.num_grids):
                    prediction = tf.reshape(prediction[..., :45], (self.num_grids, self.num_grids, 9, 5))
                    for k in range(self.num_grids):
                        for l in range(self.num_grids):
                            for m in range(9):
                                if prediction[k, l, m, 0] > self.threshold:
                                    pre_boxes.append(predictions[k, l, m, 1:5])
            for bbox in pre_boxes:
                pass

    def cal_iou(self):
        pass

    def eval(self):
        inputs = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3))
        predictions = VOC_model.Model().new_network_1(inputs)
        eval_data = data.Data(self.test_file, self.batch_size, self.image_size)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print('restoring from', tf.train.latest_checkpoint(self.save_path))
            saver.restore(sess, tf.train.latest_checkpoint(self.save_path))

            AP = []
            # Have test image 4336 in total
            for i in range(4330):
                images, labels = eval_data.load_data(flip=False)
                pre = sess.run(predictions, feed_dict={inputs, images})
                ap = self.cal_AP(labels, pre)
                AP.append(ap)
            map = sum(AP) / len(AP)
            print('mAP:', map)

    def visual(self, image, labels, result_cls, sess, threshold=0.5):
        anchor_boxes = sess.run(self.anchor_boxes())
        boxes, scores, cls = [], [], []
        for i in range(self.num_grids):
            for j in range(self.num_grids):
                k = np.where(labels[i, j, :, 0] == np.max(labels[i, j, :, 0]))
                k = k[0][0]
                # if i*19+j==104:
                if labels[i, j, k, 0] > threshold:
                    print(i * 19 + j)
                    center_x = (labels[i, j, k, 1] + i) / self.num_grids
                    center_y = (labels[i, j, k, 1] + j) / self.num_grids
                    xmin, xmax = int((center_x - labels[i, j, k, 3] / 2 * anchor_boxes[k][0]) * self.image_size), \
                                 int((center_x + labels[i, j, k, 3] / 2 * anchor_boxes[k][0]) * self.image_size)
                    ymin, ymax = int((center_y - labels[i, j, k, 4] / 2 * anchor_boxes[k][1]) * self.image_size), \
                                 int((center_y + labels[i, j, k, 4] / 2 * anchor_boxes[k][1]) * self.image_size)
                    coord = [ymin, xmin, ymax, xmax]
                    boxes.append(coord)
                    scores.append(labels[i, j, k, 0])
                    cls.append(tf.arg_max(result_cls[i, j], -1))
        try:
            truth_boxes = tf.image.non_max_suppression(np.array(boxes), np.array(scores), 10, 0.3)
            for i in sess.run(truth_boxes):
                # r, g, b = random.random(), random.random(), random.random()
                cv2.rectangle(image, (boxes[i][1], boxes[i][0]), (boxes[i][3], boxes[i][2]), (0, 1, 0), 1)
                for k in self.VOC_LABELS.keys():
                    if self.VOC_LABELS[k] == sess.run(cls[i]):
                        print(k)
                        cv2.putText(image, str(k)+str(scores[i])[:4], (boxes[i][1], boxes[i][0]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        except:
            print('No bbox')
        return image


    def submission(self):
        pass



