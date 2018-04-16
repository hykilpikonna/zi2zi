# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
from PIL import ImageFont

from model.preprocessing_helper import draw_single_char, get_textsize, save_imgs
from model.unet import UNet
from model.utils import merge, scale_back

"""
People are made to have fun and be 中二 sometimes
                                --Bored Yan LeCun
"""

parser = argparse.ArgumentParser(description='Inference for unseen data')
parser.add_argument('--model_dir', dest='model_dir', default="experiments_finetune/checkpoint/experiment_0_batch_32",
                    help='directory that saves the model checkpoints')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='number of examples in batch')
parser.add_argument('--text', type=str, default="人生是一条马尔可夫链", help='the source images for inference')
parser.add_argument('--embedding_id', type=int, default=48, help='embeddings involved')
parser.add_argument('--save_dir', default='save_dir', type=str, help='path to save inferred images')
parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
                    help='use conditional instance normalization in your model')
parser.add_argument('--char_size', dest='char_size', type=int, default=150, help='character size')
parser.add_argument('--src_font', dest='src_font', default='data/raw_fonts/SimSun.ttf', help='path of the source font')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')

args = parser.parse_args()


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    src_font = ImageFont.truetype(args.src_font, size=args.char_size)
    src_pix_size = get_textsize(src_font)

    with tf.Session(config=config) as sess:
        model = UNet(batch_size=args.batch_size)
        model.register_session(sess)
        model.build_model(is_training=False, inst_norm=args.inst_norm)
        model.load_model(args.model_dir)

        count = 0
        batch_buffer = list()
        source_imgs = []
        for ch in list(args.text):
            src_img = draw_single_char(ch, src_font, args.canvas_size, src_pix_size)
            np_src_img = np.array(src_img, dtype=np.float32) / 255.
            np_pair_src_img = np.concatenate([np_src_img, np_src_img], axis=2)  # 256,256,6
            source_imgs.append(np_pair_src_img)

        fake_imgs = model.generate_fake_samples(source_imgs, [args.embedding_id] * len(source_imgs))[0]
        merged_fake_images = merge(scale_back(fake_imgs), [model.batch_size, 1])  # scale 0-1
        batch_buffer.append(merged_fake_images)
        if len(batch_buffer) == 10:
            save_imgs(batch_buffer, count, args.save_dir)
            batch_buffer = list()
        count += 1

        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, count, args.save_dir)


if __name__ == '__main__':
    tf.app.run()
