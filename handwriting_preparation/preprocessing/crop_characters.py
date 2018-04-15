import os
import pdb

import PIL
from PIL import Image, ImageEnhance


def draw_single_char(img, canvas_size, factor=1.):
    font_pix_size = int(max(img.size) * factor)
    bg_img = Image.new("RGB", (font_pix_size, font_pix_size), (255, 255, 255))
    offset = ((font_pix_size - img.size[0]) // 2, (font_pix_size - img.size[1]) // 2)
    bg_img.paste(img, offset)
    return bg_img.resize((canvas_size, canvas_size), resample=PIL.Image.LANCZOS)


image_path = "../images/test_image.jpg"
box_path = "../images/test_image.box"
out_dir = "../characters/"

# Load the original image:
img = Image.open(image_path)

# makedir
try:
    os.makedirs(out_dir)
except:
    pass

n = 0
with open(box_path, "r") as f:
    for line in f:
        if n >= 156: break  # custom rules, so that you don't have fix all the box results

        char, x1, y1, x2, y2, _ = line.rstrip().split(' ')
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Crop the character based on Box result
        char_img = img.crop((x1, img.size[1] - y2, x2, img.size[1] - y1))

        # Leave enough space and resize to canvas_size
        char_img = draw_single_char(char_img, canvas_size=256, factor=1.1)

        # Add brightness
        contrast = ImageEnhance.Contrast(char_img)
        char_img = contrast.enhance(2.)

        # Add brightness
        brightness = ImageEnhance.Brightness(char_img)
        char_img = brightness.enhance(2.)

        char_img.save(os.path.join(out_dir, char + ".jpg"), "JPEG", quality=100)
        n += 1
