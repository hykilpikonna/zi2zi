# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import argparse
import importlib
import json
import os
import pdb
import sys

import collections
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# reload(sys)
# sys.setdefaultencoding("utf-8")
importlib.reload(sys)

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None
GB775_CHARSET = None
GB6763_CHARSET = None

DEFAULT_CHARSET = "./charset/cjk_cn.json"


def load_global_charset():
    global CN_CHARSET, JP_CHARSET, KR_CHARSET, CN_T_CHARSET, GB775_CHARSET, GB6763_CHARSET
    cjk = json.load(open(DEFAULT_CHARSET))
    CN_CHARSET = cjk["gbk"]
    JP_CHARSET = cjk["jp"]
    KR_CHARSET = cjk["kr"]
    CN_T_CHARSET = cjk["gb2312_t"]
    GB775_CHARSET = cjk["gb775"]
    GB6763_CHARSET = cjk["gb6763"]


# def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
#     img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
#     draw = ImageDraw.Draw(img)
#
#     draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font)
#     return img


def get_offset_size(ch, draw, font, font_pix_size):
    pix_size = draw.textsize(ch, font=font)
    x_offset = (font_pix_size - pix_size[0]) // 2
    y_offset = (font_pix_size - pix_size[1]) // 2
    return x_offset, y_offset


def draw_single_char(ch, font, canvas_size, font_pix_size=160):
    img = Image.new("RGB", (font_pix_size, font_pix_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, fill=(0, 0, 0), font=font)
    return img.resize((canvas_size, canvas_size))


def draw_example(ch, src_font, dst_font, canvas_size, filter_hashes, src_pix_size, dst_pix_size):
    src_img = draw_single_char(ch, src_font, canvas_size, src_pix_size)
    dst_img = draw_single_char(ch, dst_font, canvas_size, dst_pix_size)

    # check the filter example in the hashes or not
    dst_hash = hash(dst_img.tobytes())
    if dst_hash in filter_hashes or np.min(src_img) == 255 or np.min(dst_img) == 255:
        return None

    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img


def filter_recurring_hash(charset, font, canvas_size, font_pix_size):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    """
    _charset = charset[:]
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size, font_pix_size)
        hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]


def get_textsize(font):
    # size = int(150*1.0)
    img = Image.new("RGB", (1, 1), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # draw.text((0,0), '器', fill=(0, 0, 0), font=font)
    # img.show()
    # pdb.set_trace()
    char_size = draw.textsize('早', font=font)

    return int(max(char_size))


def font2img(src, dst, charset, char_size, canvas_size,
             sample_count, sample_dir, label=0, filter_by_hash=True):
    assert os.path.isfile(src), "src file doesn't exist:%s" % src
    assert os.path.isfile(dst), "dst file doesn't exist:%s" % dst

    src_font = ImageFont.truetype(src, size=char_size)
    src_pix_size = get_textsize(src_font)
    dst_font = ImageFont.truetype(dst, size=char_size)
    dst_pix_size = get_textsize(dst_font)

    filter_hashes = set()
    if filter_by_hash:
        filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, dst_pix_size))
        print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

    count = 0

    for c in charset:
        if count == sample_count:
            break
        e = draw_example(c, src_font, dst_font, canvas_size, filter_hashes, src_pix_size, dst_pix_size)
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
            count += 1
            if count % 100 == 0:
                print("processed %d chars" % count)


load_global_charset()
parser = argparse.ArgumentParser(description='Convert font to images')
# parser.add_argument('--src_font', dest='src_font', required=True, help='path of the source font')
# parser.add_argument('--dst_font', dest='dst_font', required=True, help='path of the target font')
parser.add_argument('--filter', dest='filter', type=int, default=0, help='filter recurring characters')
parser.add_argument('--charset', dest='charset', type=str, default='CN',
                    help='charset, can be either: CN, JP, KR , GB775, GB6763 or a one line file')
parser.add_argument('--shuffle', dest='shuffle', type=int, default=True, help='shuffle a charset before processings')
parser.add_argument('--char_size', dest='char_size', type=int, default=150, help='character size')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--sample_count', dest='sample_count', type=int, default=500, help='number of characters to draw')
parser.add_argument('--sample_dir', dest='sample_dir', help='directory to save examples')

args = parser.parse_args()

# img = Image.new("RGB", (150, 200), (255, 255, 255))
# draw = ImageDraw.Draw(img)
# src_font = ImageFont.truetype('data/raw_fonts/井柏然体.ttf', size=150)
# print(get_textsize(src_font))
# draw.text((0, 0), '吖', fill=(0, 0, 0), font=src_font)
# pdb.set_trace()

if __name__ == "__main__":

    args.src_font = "data/raw_fonts/simsun.ttf"
    args.fonts_dir = "data/raw_fonts"
    args.sample_dir = "data/paired_images"

    label = 0
    for root, dirs, files in os.walk(args.fonts_dir):
        for name in files:
            if name.lower().endswith(".ttf") and name.lower() not in ["simsun.ttf", "井柏然体.ttf"]:
                dst_font = os.path.join(root, name)

                if args.charset in ['CN', 'JP', 'KR', 'CN_T', 'GB775', 'GB6763']:
                    charset = locals().get("%s_CHARSET" % args.charset)
                else:
                    charset = [c for c in open(args.charset).readline()[:-1].decode("utf-8")]

                if args.shuffle:
                    np.random.shuffle(charset)
                font2img(args.src_font, dst_font, charset, args.char_size,
                         args.canvas_size,
                         args.sample_count, args.sample_dir, label, args.filter)
                label += 1
