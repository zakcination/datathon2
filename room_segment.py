import cv2
import numpy as np

def segment_rooms(image):
    red_lower, red_upper = 70, 85
    green_lower, green_upper = 100, 115
    blue_lower, blue_upper = 170, 185

    red_lower1, red_upper1 = 0, 10
    green_lower1, green_upper1 = 70, 85
    blue_lower1, blue_upper1 = 145, 165

    mask = cv2.inRange(image, (blue_lower, green_lower, red_lower), (blue_upper, green_upper, red_upper))
    mask1 = cv2.inRange(image, (blue_lower1, green_lower1, red_lower1), (blue_upper1, green_upper1, red_upper1))

    mask = cv2.bitwise_or(mask, mask1)

    image[mask == 255] = (255, 255, 255)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 7), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    dst = cv2.cornerHarris(opening ,2,3,0.04)
    dst = cv2.dilate(dst, None) 

    corners = dst > 0.01 * dst.max()

    for y,row in enumerate(corners):
        x_same_y = np.argwhere(row)
        for x1, x2 in zip(x_same_y[:-1], x_same_y[1:]):
            if x2[0] - x1[0] < 190:  # max line length
                color = 0
                # print(x1)
                cv2.line(opening, (x1[0], y), (x2[0], y), 255, 1)

        for x,col in enumerate(corners.T):
            y_same_x = np.argwhere(col)
            for y1, y2 in zip(y_same_x[:-1], y_same_x[1:]):
                if y2[0] - y1[0] < 190:
                    color = 0
                    # print(y1)
                    cv2.line(opening, (x, y1[0]), (x, y2[0]), 255, 1)

    num_labels, labels = cv2.connectedComponents(255-opening)
    masks = [labels == i for i in range(1, num_labels + 1)]

    contours = [cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for mask in masks]

    areas = []
    image_area = cv2.contourArea(contours[0][0])
    filtered_contours = []

    for cnt in contours[1:]: # first is whole image
        # print(len(cnt))
        if len(cnt) > 0:
            area = cv2.contourArea(cnt[0]) + 0.0000001
            area_ratio = image_area / area
            if area_ratio < 800:
                areas.append(area)
                filtered_contours.append(cnt)
                print("Contour area:", area)
                cv2.drawContours(image, cnt, -1, (0, 255, 0), 2)

    return filtered_contours, image