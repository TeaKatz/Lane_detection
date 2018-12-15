import config as cf
from libs.image_processing import get_edge, crop_image, apply_threshold_bgr
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process(image):
    image = crop_image(image, (500, 0), (image.shape[0], image.shape[1]))
    image = apply_threshold_bgr(image, (175, 195), (142, 155), (120, 133))
    #image = get_edge(image)

    return image


def show_image(image):
    cv2.imshow(cf.WINDOW_NAME, image)

    while True:
        if cv2.waitKey(25) & 0xFF == 27:    #Press ESC
            cv2.destroyAllWindows()
            break
        elif cv2.getWindowProperty(cf.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:   #Close window
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    if len(sys.argv) == 2:
        img_dir = str(sys.argv[1])
        if os.path.isfile(img_dir):
            image = cv2.imread(img_dir)
            image = cv2.resize(image, cf.WINDOW_SIZE)
            cv2.imshow("original", image)

            image = process(image)

            show_image(image)
        else:
            sys.exit("File does not exits.")
    else:
        sys.exit("Please define image directory.")