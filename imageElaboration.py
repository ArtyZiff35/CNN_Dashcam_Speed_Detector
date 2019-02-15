import numpy as np
import cv2
from utilities import *


# This function applies all elaboration steps to the image
def elaborateImage(newFrame):
    # Adjusting brightness and contrast
    newFrameAdjusted = apply_brightness_contrast(newFrame, 90, 90)

    # Threshold so that only yellow and white are kept. Result is greyscale
    newFrameThreshold = thresholdWhiteAndYellow(newFrameAdjusted)

    # Apply Gaussian blur to reduce noise
    newFrameBlurred = cv2.GaussianBlur(newFrameThreshold, (5, 5), 0)

    # Applying canny edge detection
    newFrameEdges = cv2.Canny(newFrameBlurred, 100, 200)

    # Cutting a region of interest
    height, width = newFrameEdges.shape
    # Creating white polygonal shape on black image
    bottomLeft = [10, height - 130]
    topLeft = [width / 3 + 60, height / 2]
    topRight = [width * 2 / 3 - 60, height / 2]
    bottomRight = [width - 10, height - 130]
    pts = np.array([bottomLeft, topLeft, topRight, bottomRight], np.int32)
    pts = pts.reshape((-1, 1, 2))
    blackImage = np.zeros((height, width, 1), np.uint8)
    polygonalShape = cv2.fillPoly(blackImage, [pts], (255, 255, 255))
    # Doing AND operation with newFrameEdges
    newFrameROI = cv2.bitwise_and(newFrameEdges, newFrameEdges, mask=polygonalShape)

    # Hough transform to detect straight lines. Returns an array of r and theta values
    lines = cv2.HoughLinesP(newFrameROI, 1, np.pi / 180, 15)
    blackImage = np.zeros((height, width, 1), np.uint8)
    newFrameHough = drawHoughTransformLines(blackImage, lines)

    # Drawing road from original frame
    newFrameGrey = cv2.cvtColor(newFrameAdjusted, cv2.COLOR_BGR2GRAY)
    coloredMaskedRoad = cv2.bitwise_and(newFrameGrey, newFrameGrey, mask=polygonalShape)
    newFrameMaskAndRoad = cv2.add(coloredMaskedRoad, newFrameROI)   # Adding canny edge overlay to highlight the lane markers

    # Cutting image basing on mask size
    result = cutTopAndBottom(coloredMaskedRoad, int(height / 2), int(height - 130))

    return result