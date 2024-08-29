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

def get_lane_lines(image, avg):
    if avg is None:
        return None

    left_lane, right_lane = avg
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)

    return left_line, right_line

def draw_lane_lines(image, lines):
    if lines is None:
        return image

    line_image = np.zeros_like(image)

    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color=[0, 255, 0], thickness=2)

    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def draw_steering_line(image, angle):
    angle *= 5
    line_len = 0.05 * w

    line_image = np.zeros_like(image)
    cv2.arrowedLine(line_image, (int(w / 2), int(h / 2)), (int(w / 2 - line_len * sin(angle)), int(h / 2 - line_len * cos(angle))), color=[0, 255, 0], thickness=2, tipLength=0.5)

    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def transform(point):
    neg_pitch = 0.12 * pi
    cam_height = w
    focal_len= 0.5 * w

    x, y = point
    x -= cam_height / 2
    y = cam_height / 2 - y

    theta = neg_pitch - atan(y / focal_len)
    cot = cos(theta) / sin(theta)
    y_ret = cam_height * cot
    l = sqrt(cam_height ** 2 + y_ret ** 2)
    x_ret = l * x / focal_len

    return x_ret, y_ret

def angle(p0, p1):
    x0, y0 = p0
    x1, y1 = p1

    return atan2(y1 - y0, x1 - x0)

def distance(p0, p1):
    x0, y0 = p0
    x1, y1 = p1

    return sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

# exponential smoothening over viewport lane lines
slope_alpha = 0.9
intercept_alpha = 0.9
avg_slope_intercepts = None

# linear regression over mean ground lane angle
m = 1
c = m * 0.13

pred = []
angle_diff = []
lane_dist = []

#out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 90, (w, h))

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

    l_angle = m * (angle(tf_l_p0, tf_l_p1) - pi / 2) + c
    r_angle = m * (angle(tf_r_p0, tf_r_p1) - pi / 2) + c

    pred_angle = (l_angle + r_angle) / 2
    pred.append(pred_angle)

    angle_diff.append(l_angle - r_angle)

    d0 = distance(tf_l_p0, tf_r_p0)
    d1 = distance(tf_l_p1, tf_r_p1)
    avg_d = (d0 + d1) / 2
    lane_dist.append(avg_d * cos(abs(pred_angle)) / (w * 10))

    lane_lines = get_lane_lines(masked, avg_slope_intercepts)
    res = draw_steering_line(draw_lane_lines(resized, lane_lines), pred_angle)

    cv2.imshow('frame', res)
    #out.write(res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

fig, ax = plt.subplots()
ax.plot(range(len(pred)), pred, label='avg angle')
ax.plot(range(len(angle_diff)), angle_diff, label='angle diff')
ax.plot(range(len(lane_dist)), lane_dist, label='lane dist')

ax.legend()
plt.title('lane regression data')
plt.show()
