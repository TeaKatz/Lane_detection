import numpy as np
import cv2

def FindEdge(image, kernel_size=5, threshold_low=100, threshold_high=200):
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    image = cv2.Canny(image, threshold_low, threshold_high)
    return image


def region_of_interest(image):
    height = image.shape[0]
    weight = image.shape[1]
    polygon = np.array([[(int(weight * 0), height),
                         (int(weight * 0.45), int(height * 0.65)),
                         (int(weight * 0.55), int(height * 0.65)),
                         (int(weight * 1), height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    weight = image.shape[1]
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            slope = (y2 - y1) / (x2 - x1 + 1e-7)
            if slope > 0.1:
                if x1 > int(weight * 0.55):
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
            elif slope < -0.1:
                if x1 < int(weight * 0.45):
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return line_image


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    weight = image.shape[1]
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < -0.1 and slope > -1:
                if x1 < int(weight * 0.45):
                    left_fit.append((slope, intercept))
            elif slope > 0.1 and slope < 1:
                if x1 > int(weight * 0.55):
                    right_fit.append((slope, intercept))

        lines = []
        if left_fit:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = make_coordinates(image, left_fit_average)
            lines.append(left_line)

        if right_fit:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = make_coordinates(image, right_fit_average)
            lines.append(right_line)

        return np.array(lines)
    else:
        return lines


def lane_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = FindEdge(gray)
    crop = region_of_interest(edge)
    lines = cv2.HoughLinesP(crop, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, lines)
    return line_image


if __name__ == "__main__":
    video = cv2.VideoCapture("inputs/video01.mp4")

    while video.isOpened():
        is_capturing, frame = video.read()

        if is_capturing:
            frame = cv2.resize(frame, None, fx=1, fy=1)

            line_image = lane_detection(frame)
            final_image = cv2.addWeighted(line_image, 0.8, frame, 1, 0)

            cv2.imshow("frame", final_image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()
