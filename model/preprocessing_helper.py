import PIL
import numpy as np
from PIL import Image
from PIL import ImageDraw
import os
from .utils import save_concat_images

def get_offset_size(ch, draw, font, font_pix_size):
    pix_size = draw.textsize(ch, font=font)
    x_offset = (font_pix_size - pix_size[0]) // 2
    y_offset = (font_pix_size - pix_size[1]) // 2
    return x_offset, y_offset


def get_textsize(font):
    # size = int(150*1.0)
    img = Image.new("RGB", (1, 1), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # draw.text((0,0), '器', fill=(0, 0, 0), font=font)
    # img.show()
    # pdb.set_trace()
    char_size = draw.textsize('早', font=font)

    return int(max(char_size))


def save_imgs(imgs, count, save_dir):
    p = os.path.join(save_dir, "inferred_%04d.png" % count)
    save_concat_images(imgs, img_path=p)
    print("generated images saved at %s" % p)


def draw_single_char(ch, font, canvas_size, font_pix_size=160):
    img = Image.new("RGB", (font_pix_size, font_pix_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, fill=(0, 0, 0), font=font)
    return img.resize((canvas_size, canvas_size), resample=PIL.Image.LANCZOS)


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
