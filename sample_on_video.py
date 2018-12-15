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


if __name__ == "__main__":
    if len(sys.argv) == 2:
        img_dir = str(sys.argv[1])
        if os.path.isfile(img_dir):
            video = cv2.VideoCapture(img_dir)
            while video.isOpened():
                is_capturing, frame = video.read()
                if is_capturing:
                    frame = cv2.resize(frame, cf.WINDOW_SIZE)
                    cv2.imshow("original", frame)

                    frame = process(frame)

                    cv2.imshow(cf.WINDOW_NAME, frame)

                    if cv2.waitKey(25) & 0xFF == 27:  # Press ESC
                        break
                    elif cv2.getWindowProperty(cf.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:  # Close window
                        break
                else:
                    break
            video.release()
            cv2.destroyAllWindows()
        else:
            sys.exit("File does not exits.")
    else:
        sys.exit("Please define video directory.")
