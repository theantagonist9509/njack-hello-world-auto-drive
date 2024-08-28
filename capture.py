from math import *
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

#if (len(sys.argv) != 2):
#    print('Error: Invalid arguments')
#    sys.exit(1)
#
#addr = sys.argv[1]
#cap = cv2.VideoCapture('rtsp://' + addr + '/h264_ulaw.sdp')
cap = cv2.VideoCapture('test.mkv')

h = 540
w = 960

def region_select(image):
    mask = np.zeros_like(image)   

    rows, cols = image.shape[:2]
    #bottom_left  = [0, rows]
    #top_left     = [0, rows * 0.5]
    #bottom_right = [cols, rows]
    #top_right    = [cols, rows * 0.5]
    bottom_left  = [0.25 * cols, rows]
    top_left = [0.45 * cols, 0.65 * rows]
    bottom_right     = [0.8 * cols, rows]
    top_right    = [0.65 * cols, 0.65 * rows]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, 255)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def hough_transform(image):
    rho = 1             
    theta = pi / 180   
    threshold = 20      
    minLineLength = 20  
    maxLineGap = 500    
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
                           minLineLength = minLineLength, maxLineGap = maxLineGap)

def average_slope_intercept(lines):
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)

    if lines is None:
        return None
     
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))

            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    if left_weights == [] or right_weights == []:
        return None

    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights)

    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None

    slope, intercept = line

    if slope == 0:
        return None

    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, avg):
    if avg is None:
        return None

    left_lane, right_lane = avg
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)

    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    if lines is None:
        return image

    line_image = np.zeros_like(image)

    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness=thickness)

    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def transform(point):
    neg_pitch = 0.13 * pi
    h = 1000
    f = 0.5 * w

    x, y = point
    x -= h / 2
    y = h / 2 - y

    theta = neg_pitch - atan(y / f)
    cot = cos(theta) / sin(theta)
    y_ret = h * cot
    l = sqrt(h * h + y_ret * y_ret)
    x_ret = l * x / f

    return x_ret, y_ret

def angle(p0, p1):
    x0, y0 = p0
    x1, y1 = p1

    return atan2(y1 - y0, x1 - x0)

slope_alpha = 0.9
intercept_alpha = 0.9
avg_slope_intercepts = None

pred = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (5, 5))
    edges = cv2.Canny(blurred, 75, 125)
    masked = region_select(edges)
    hough = hough_transform(masked)
    new_avg_slope_intercepts = average_slope_intercept(hough)

    if new_avg_slope_intercepts is None:
        continue

    if avg_slope_intercepts is None:
        m_l = new_avg_slope_intercepts[0][0]
        c_l = new_avg_slope_intercepts[0][1]
        m_r = new_avg_slope_intercepts[1][0]
        c_r = new_avg_slope_intercepts[1][1]

        avg_slope_intercepts = new_avg_slope_intercepts
    else:
        m_l = slope_alpha * avg_slope_intercepts[0][0] + (1 - slope_alpha) * new_avg_slope_intercepts[0][0]
        c_l = intercept_alpha * avg_slope_intercepts[0][1] + (1 - intercept_alpha) * new_avg_slope_intercepts[0][1]
        m_r = slope_alpha * avg_slope_intercepts[1][0] + (1 - slope_alpha) * new_avg_slope_intercepts[1][0]
        c_r = intercept_alpha * avg_slope_intercepts[1][1] + (1 - intercept_alpha) * new_avg_slope_intercepts[1][1]

        avg_slope_intercepts = ((m_l, c_l), (m_r, c_r))

    l_p0 = ((h - c_l) / m_l, h)
    l_p1 = ((0.75 * h - c_l) / m_l, 0.75 * h)
    r_p0 = ((h - c_r) / m_r, h)
    r_p1 = ((0.75 * h - c_r) / m_r, 0.75 * h)

    tf_l_p0 = transform(l_p0)
    tf_l_p1 = transform(l_p1)
    tf_r_p0 = transform(r_p0)
    tf_r_p1 = transform(r_p1)

    pred.append(angle(tf_l_p0, tf_l_p1) + angle(tf_r_p0, tf_r_p1) - pi)
    print(str(angle(tf_l_p0, tf_l_p1)) + ', ' + str(angle(tf_r_p0, tf_r_p1)))
    #print(str(tf_l_p0) + str(tf_l_p1))

    lines = lane_lines(masked, avg_slope_intercepts)
    res = draw_lane_lines(gray, lines)

    cv2.imshow('masked', masked)
    cv2.imshow('frame', res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

fig, ax = plt.subplots()
ax.plot(range(len(pred)), pred)

plt.show()
