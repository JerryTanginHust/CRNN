#! -*- encoding: utf-8 -*-
import os
from argparse import ArgumentParser
from network.config import configure as cfg
from network.utils.coder import strLabelConverter
from network.utils.eval import ctc_label
from network.network import CRNN
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import shutil
import editdistance
import xml.dom.minidom
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


parser = ArgumentParser(description='crnn-demo')
parser.add_argument('--base_net','-b',default='res',type=str)
parser.add_argument("--lexicon", type=str, default="./experiments/demo/en.lexicon", help="")
parser.add_argument('--image_root', '-i', default='./data', type=str)
parser.add_argument('--tags_file', '-g', default='./data/test_data_1k.tags', type=str)
parser.add_argument('--result_save_path', '-r', default='result.txt', type=str)
parser.add_argument('--model_dir', '-m', default='./experiments/demo/ckpt/crnn_res_iter_176000.ckpt', type=str)
args = parser.parse_args()


def load_image(img_path):
    img = Image.open(img_path)
    img = img.convert('L')
    w, h = img.size
    rio = h/32.0
    w0 = int(round(w/rio))
    img = img.resize((w0, 32),Image.ANTIALIAS)
    img = np.array(img)
    img = np.reshape(img, (1,32, w0, 1))
    img = img.astype(np.float32)/128.0 - 1.0
    return img

dictionary = args.lexicon
with open(dictionary) as f:
    lines = f.readlines()
alphabet = []
for line in lines:
    tmp = line.strip('\n')
    alphabet.append(tmp)
alpha = ('').join(alphabet)
converter = strLabelConverter(alpha)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
    net = CRNN(cfg.IMG_SHAPE[0], head=args.base_net,batch_size= 1,istrain=False)
    img = tf.placeholder(tf.float32, shape=(1,32, None, 1))
    _, _, logits = net(img)
    logits = tf.nn.softmax(logits,dim=-1)
    out_max = tf.reduce_max(logits,axis=2)
    score = tf.squeeze(tf.reduce_mean(out_max,axis=0))
    preds = tf.argmax(logits, axis=2)  # t,b
    preds = tf.transpose(preds, perm=(1, 0))  # b,t
    preds = tf.squeeze(preds)
    saver = tf.train.Saver()

    print("restore from %s" %(args.model_dir))
    saver.restore(sess, args.model_dir)
    print("restore success")

    num_total, num_correct = 0, 0
    edit_distance_accuracies = []
    result_fo = open(args.result_save_path, "w")
    with open(args.tags_file) as fo:
        for line in fo:
            image_path, gt = line.rstrip("\n").split(" ", 1)
            image_path = os.path.join(args.image_root, image_path)
            pred_val, score_val = sess.run([preds, score], feed_dict={img: load_image(image_path)})
            text = converter.decode_txt(ctc_label(pred_val))
            write_buf = "{}: {} => {}".format(image_path, gt, text)
            result_fo.write(write_buf + "\n")
            print(write_buf)

            num_total += 1
            if gt == text:
                num_correct += 1
            ed = editdistance.eval(gt, text)
            edit_distance_accuracies.append(1 - ed / len(gt))
    print("Word Accuracy: {}/{}={:.3}%".format(num_correct, num_total, num_correct * 100 / num_total))
    print("Edit Accuracy: {}".format(sum(edit_distance_accuracies)/len(edit_distance_accuracies)))

