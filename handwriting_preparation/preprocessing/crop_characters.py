import os

import PIL
from PIL import Image, ImageEnhance

from model.preprocessing_helper import CANVAS_SIZE, CHAR_SIZE


def draw_single_char(img, canvas_size):
    width, height = img.size
    factor = width * 1.0 / CHAR_SIZE

    max_height = CANVAS_SIZE + 30
    if height / factor > max_height:  # too long
        img = img.crop((0, 0, width, int(max_height * factor)))
    if height / factor > CHAR_SIZE + 5:  # CANVAS_SIZE/CHAR_SIZE is a benchmark, height should be less
        factor = height * 1.0 / CHAR_SIZE

    img = img.resize((int(width / factor), int(height / factor)), resample=PIL.Image.LANCZOS)

    bg_img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    offset = ((canvas_size - img.size[0]) // 2, (canvas_size - img.size[1]) // 2)
    bg_img.paste(img, offset)
    return bg_img


def char_img_iter(image_path, box_path):
    assert os.path.isfile(image_path), "image file doesn't exist: %s" % image_path
    assert os.path.isfile(box_path), "image box file doesn't exist %s" % box_path

    n = 0
    img = Image.open(image_path)
    with open(box_path, "r") as f:
        for line in f:
            if n >= 156 and "test_image.jpg" in image_path: break  # custom rules, so that you don't have fix all the box results

            ch, x1, y1, x2, y2, _ = line.rstrip().split(' ')
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # Crop the character based on Box result
            char_img = img.crop((x1, img.size[1] - y2, x2, img.size[1] - y1))

            # Leave enough space and resize to canvas_size
            char_img = draw_single_char(char_img, canvas_size=CANVAS_SIZE)

            # Add brightness
            contrast = ImageEnhance.Contrast(char_img)
            char_img = contrast.enhance(2.)

            # Add brightness
            brightness = ImageEnhance.Brightness(char_img)
            char_img = brightness.enhance(2.)

            yield ch, char_img
            n += 1


if __name__ == '__main__':
    image_path = "../images/test_image.jpg"
    box_path = "../images/test_image.box"
    out_dir = "../characters/test_image/"

    # makedir
    try:
        os.makedirs(out_dir)
    except:
        pass

    # For debug only
    for ch, char_img in char_img_iter(image_path, box_path):
        char_img.save(os.path.join(out_dir, ch + ".jpg"), "JPEG", quality=100)
