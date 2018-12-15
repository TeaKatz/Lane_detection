import sys
import cv2
import numpy as np

def get_edge(image, kernel_size=5, threshold_low=100, threshold_high=200):
    #Precondition
    if len(image.shape) != 2:    #Check if input is not gray scale?
        if len(image.shape) == 3:    #Check if input dimension is correct?
            if image.shape[2] == 3:     # Check if number of chanel is correct?
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] != 1:
                sys.exit("Expect input as BGR or GRAY scale, but get input of shape {0}.".format(image.shape))
        else:
            sys.exit("Expect input as BGR or GRAY scale, but get input of shape {0}.".format(image.shape))

    if not isinstance(kernel_size, int):
        sys.exit("kernel_size must be integer, but get {0}.".format(kernel_size))
    elif kernel_size % 2 == 0:
        sys.exit("kernel_size should be odd number, but get {0}.".format(kernel_size))

    if not isinstance(threshold_low, int):
        sys.exit("threshold_low must be integer, but get {0}.".format(threshold_low))

    if not isinstance(threshold_high, int):
        sys.exit("threshold_high must be integer, but get {0}.".format(threshold_high))

    if threshold_low > threshold_high:
        sys.exit("threshold_low must less than threshold_high.")

    #Process
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    image = cv2.Canny(image, threshold_low, threshold_high)

    return image


def crop_image(image, pos1, pos2):
    #Precondition
    if len(image.shape) != 2:    #Check if input is not gray scale?
        if len(image.shape) == 3:    #Check if input dimension is correct?
            if image.shape[2] != 1 and image.shape[2] != 3:     # Check if number of chanel is correct?
                sys.exit("Expect input as BGR or GRAY scale, but get input of shape {0}.".format(image.shape))
        else:
            sys.exit("Expect input as BGR or GRAY scale, but get input of shape {0}.".format(image.shape))

    if np.shape(pos1) != (2,):
        sys.exit("Expect pos1 as tuples (y, x), but get {0}".format(pos1))

    if np.shape(pos2) != (2,):
        sys.exit("Expect pos1 as tuples (y, x), but get {0}".format(pos2))

    # Process
    y1, x1 = pos1
    y2, x2 = pos2
    min_x = min(x1, x2)
    max_x = max(x1, x2)
    min_y = min(y1, y2)
    max_y = max(y1, y2)
    image = image[min_y:max_y, min_x:max_x]

    return image


def apply_threshold_bgr(image, threshold_b, threshold_g, threshold_r):
    #Precondition
    if len(image.shape) != 2:    #Check if input is not gray scale?
        if len(image.shape) == 3:    #Check if input dimension is correct?
            if image.shape[2] != 1 and image.shape[2] != 3:     # Check if number of chanel is correct?
                sys.exit("Expect input as BGR or GRAY scale, but get input of shape {0}.".format(image.shape))
        else:
            sys.exit("Expect input as BGR or GRAY scale, but get input of shape {0}.".format(image.shape))

    if np.shape(threshold_b) != (2,):
        sys.exit("Expect threshold_b as tuples (low, high), but get {0}".format(threshold_b))
    elif threshold_b[0] > threshold_b[1]:
        sys.exit("Expect threshold_b as tuples (low, high), but get {0}".format(threshold_b))

    if np.shape(threshold_g) != (2,):
        sys.exit("Expect threshold_b as tuples (low, high), but get {0}".format(threshold_g))
    elif threshold_g[0] > threshold_g[1]:
        sys.exit("Expect threshold_b as tuples (low, high), but get {0}".format(threshold_g))

    if np.shape(threshold_r) != (2,):
        sys.exit("Expect threshold_b as tuples (low, high), but get {0}".format(threshold_r))
    elif threshold_r[0] > threshold_r[1]:
        sys.exit("Expect threshold_b as tuples (low, high), but get {0}".format(threshold_r))

    #Process
    min_b, max_b = threshold_b
    min_g, max_g = threshold_g
    min_r, max_r = threshold_r
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    _, b = cv2.threshold(b, min_b, max_b, cv2.THRESH_BINARY)
    _, g = cv2.threshold(g, min_g, max_g, cv2.THRESH_BINARY)
    _, r = cv2.threshold(r, min_r, max_r, cv2.THRESH_BINARY)
    b[b != 0] = 255
    g[g != 0] = 255
    r[r != 0] = 255
    image  = b | g | r

    return image