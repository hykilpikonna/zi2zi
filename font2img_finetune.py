import os
import pdb

import numpy as np
from PIL import Image, ImageFont

from model.preprocessing_helper import draw_single_char, CHAR_SIZE, CANVAS_SIZE

src = "data/raw_fonts/SimSun.ttf"
dst_font_img_dir = "handwriting_preparation/characters/test_image/"
sample_dir = "data/paired_images_finetune"
label = 47
resample = 1


def draw_example(ch, src_font, dst_img, canvas_size):
    src_img = draw_single_char(ch, src_font, canvas_size)

    assert dst_img.size == (canvas_size, canvas_size), pdb.set_trace()

    if np.min(src_img) == 255 or np.min(dst_img) == 255:
        return None

    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img


if __name__ == '__main__':
    assert os.path.isfile(src), "src file doesn't exist:%s" % src
    try:
        os.makedirs(sample_dir)
    except:
        pass

    src_font = ImageFont.truetype(src, size=CHAR_SIZE)
    count = 0
    for root, dirs, files in os.walk(dst_font_img_dir):
        for name in files:
            if name.endswith(".jpg"):
                ch = name.rsplit(".jpg", 1)[0]
                dst_img = Image.open(os.path.join(root, name))
                e = draw_example(ch, src_font, dst_img, CANVAS_SIZE)
                if e:
                    for _ in range(resample):
                        e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
                        count += 1
                        if count % 100 == 0:
                            print("processed %d chars" % count)
