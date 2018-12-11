# /home/lwu/crnn_tf/data/lexicon_fix.txt

import os
from PIL import Image
from network.utils.coder import strLabelConverter
import threading
import tensorflow as tf
import numpy as np
import logging
import cv2
logging.basicConfig(filename='logger.log', level=logging.INFO)


image_root = "./data"
lexicon_file = "experiments/demo/en.lexicon"
lexicon_list = []
with open(lexicon_file) as fo:
    for line in fo:
        lexicon_list.append(line.strip('\n'))
alphabet = ''.join(lexicon_list)
converter = strLabelConverter(alphabet)

w = 600
time_steps = w // 4


def pad(sourse):
    assert len(sourse) <= time_steps 
    for i in range(len(sourse), time_steps):
            sourse.append(0)
    return sourse


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


class CreateTFRecordThread(threading.Thread):
    """将txt标示的图片路径和标签数据，存储到tfrecord文件中"""

    def __init__(self, txt_file, tfrecord_file):
        super(CreateTFRecordThread, self).__init__()
        self.txt_file = txt_file
        self.tfrecoder_writer = tf.python_io.TFRecordWriter(tfrecord_file)

    def run(self):
        display_i = 0
        with open(self.txt_file) as fo:
            for line in fo:
                image_path, gt = line.rstrip("\n").split(" ", 1)
                try:
                    image_path = os.path.join(image_root, image_path)
                    img = Image.open(image_path)
                    self.write_example(img, gt)
                except Exception as e:
                    print(e)
                    continue
                
                if display_i % 1000 == 0:
                    print(display_i)
                    logging.info(str(display_i))
                display_i += 1

        self.tfrecoder_writer.close()

    def write_example(self, img, label):
        width, height = img.size
        img_raw = np.asarray(img.convert('L')).tostring()

        label, label_len = converter.encode_len(label)  # 将字符串标签转成整数编码
        label = pad(label)  # 将标签padding到同样的长度

        feature = {
                'height': _int64_feature(height),
                'weight': _int64_feature(width),
                'label_len': _int64_feature(label_len),
                'image_raw': _bytes_feature(img_raw),
                'label': _int64_feature(label)}
        context = tf.train.Features(feature=feature)
        example = tf.train.Example(features=context)
        self.tfrecoder_writer.write(example.SerializeToString())


def read_tfrecord(tfrecord_file, batch_size):
    """读取tfrecord文件，并生成batch数据"""
    filename_queue = tf.train.string_input_producer([tfrecord_file])
    tfrecord_reader = tf.TFRecordReader()
    _, serialized_example = tfrecord_reader.read(filename_queue)
    img_features = tf.parse_single_example(
            serialized_example, features={
                'height': tf.FixedLenFeature([], tf.int64),
                'weight': tf.FixedLenFeature([], tf.int64),
                'label_len': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([time_steps], tf.int64)})
    
    img_raw = tf.decode_raw(img_features['image_raw'], tf.uint8)
    
    img_width = tf.cast(img_features['weight'], tf.int32)
    img_height = tf.cast(img_features['height'], tf.int32)

    img_shape = tf.parallel_stack([img_height, img_width, 1])
    img = tf.reshape(img_raw, img_shape)

    img = tf.image.resize_images(img, (32, img_width))
    img = 255-img
    label_len = tf.cast(img_features['label_len'], tf.int32)
    
    img_batch, label_batch = tf.train.batch(
            [img, label_len], batch_size=batch_size, num_threads=64, capacity=2000, dynamic_pad=True)
    return img_batch, label_batch




def test_create_tfrecord():
    image_root = "./data"
    tags_file = "./data/train_data_3k.tags.sorted"
    tfrecord_file = "./data/train_data_3k.tfrecord"

    test_thread = CreateTFRecordThread(tags_file, tfrecord_file)
    test_thread.start()


if __name__ == '__main__':
    test_create_tfrecord()

