import argparse
import os
import pdb

import numpy as np
from PIL import Image, ImageFont

from handwriting_preparation.preprocessing.crop_characters import char_img_iter
from model.preprocessing_helper import draw_single_char, CHAR_SIZE, CANVAS_SIZE


def draw_example(ch, src_font, dst_img, canvas_size):
    src_img = draw_single_char(ch, src_font, canvas_size)

    assert dst_img.size == (canvas_size, canvas_size), pdb.set_trace()

    if np.min(src_img) == 255 or np.min(dst_img) == 255:
        return None

    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img


parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--src_font', default="data/raw_fonts/SimSun.ttf", help='path of the source font')
parser.add_argument('--image_basename_path',
                    default="handwriting_preparation/images/test_image",
                    help='path of the handwritten image (box file should be in the same folders)')
parser.add_argument('--embedding_id', type=int, default=67, help='embedding id')
parser.add_argument('--sample_dir', default='data/paired_images_finetune', help='directory to save examples')
parser.add_argument('--resample', type=int, default=1, help='sample with replacement')

args = parser.parse_args()

if __name__ == '__main__':
    assert os.path.isfile(args.src_font), "src file doesn't exist:%s" % args.src_font
    src_font = ImageFont.truetype(args.src_font, size=CHAR_SIZE)
    count = 0

    image_path = args.image_basename_path + ".jpg"
    box_path = args.image_basename_path + ".box"

    try:
        os.makedirs(args.sample_dir)
    except:
        pass

    for ch, dst_img in char_img_iter(image_path, box_path):
        e = draw_example(ch, src_font, dst_img, CANVAS_SIZE)
        if e:
            for _ in range(args.resample):
                e.save(os.path.join(args.sample_dir, "%d_%04d.jpg" % (args.embedding_id, count)), mode='F')
                count += 1
                if count % 100 == 0:
                    print("processed %d chars" % count)
