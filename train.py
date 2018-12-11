import os
from argparse import ArgumentParser
import easydict as ED
import tensorflow as tf
from network.config import configure as cfg
from network.train_net import solver


parser = ArgumentParser(description='crnn-ctc')
parser.add_argument('--example_root', '-e', default='600', type=str)
parser.add_argument('--base_net', '-b', default='res', type=str)
parser.add_argument('--max_iters', '-m', default=2000000, type=int)
parser.add_argument("--lexicon", type=str, default="./experiments/demo/en.lexicon", help="")
parser.add_argument("--tfrecord_path", type=str, default="./data/train_data_3k.tfrecord", help="")
parser.add_argument("--restore", type=bool, default=False, help="")
args = parser.parse_args()


imgdb = ED.EasyDict()
imgdb.path = args.tfrecord_path
max_iters = args.max_iters

with tf.Session() as sess:
    sw = solver(imgdb, args.base_net, lexicon=args.lexicon)
    print('Solving...')
    sw.train(sess, max_iters, restore=args.restore, dynamic=True)
    print('done solving')

