import numpy as np
import cv2
from utilities import *

# This function is able to highlight the contours of road lane markings using color thresholding and canny edge
def highlightRoadLaneMarkings(newFrame):
    # Adjusting brightness and contrast
    newFrameAdjusted = apply_brightness_contrast(newFrame, 100, 100)

    # Threshold so that only yellow and white are kept. Result is greyscale
    newFrameThreshold = thresholdWhiteAndYellow(newFrameAdjusted)

    # Apply Gaussian blur to reduce noise
    newFrameBlurred = cv2.GaussianBlur(newFrameThreshold, (5, 5), 0)

    # Applying canny edge detection
    newFrameEdges = cv2.Canny(newFrameBlurred, 100, 200)

    # Cutting a region of interest
    height, width = newFrameEdges.shape
    # Creating white polygonal shape on black image
    bottomLeft = [0, height - 130]
    topLeft = [width / 3 + 40, height / 2]
    topRight = [width / 3 * 2 - 40, height / 2]
    bottomRight = [width, height - 130]
    pts = np.array([bottomLeft, topLeft, topRight, bottomRight], np.int32)
    pts = pts.reshape((-1, 1, 2))
    blackImage = np.zeros((height, width, 1), np.uint8)
    polygonalShape = cv2.fillPoly(blackImage, [pts], (255, 255, 255))
    # Doing AND operation with newFrameEdges
    newFrameROI = cv2.bitwise_and(newFrameEdges, newFrameEdges, mask=polygonalShape)

    return newFrameROI

# This function applies all elaboration steps to the image
def elaborateImage(newFrame):


    # Drawing road from original frame
    newFrameAdjusted = apply_brightness_contrast(newFrame, 30, 15)
    newFrameGrey = cv2.cvtColor(newFrameAdjusted, cv2.COLOR_BGR2GRAY)
    height, width = newFrameGrey.shape
    bottomLeft = [0, height - 130]
    topLeft = [0, height / 2 + 10]
    topCenter = [width/2, height / 2 - 15]
    topRight = [width, height / 2 + 10]
    bottomRight = [width, height - 130]
    pts = np.array([bottomLeft, topLeft, topCenter, topRight, bottomRight], np.int32)
    pts = pts.reshape((-1, 1, 2))
    blackImage = np.zeros((height, width, 1), np.uint8)
    polygonalShape = cv2.fillPoly(blackImage, [pts], (255, 255, 255))
    coloredMaskedRoad = cv2.bitwise_and(newFrameGrey, newFrameGrey, mask=polygonalShape)
    #coloredMaskedRoad = cv2.equalizeHist(coloredMaskedRoad)
    newFrameROI = highlightRoadLaneMarkings(newFrame)
    newFrameMaskAndRoad = cv2.add(coloredMaskedRoad, newFrameROI)   # Adding canny edge overlay to highlight the lane markers

    # Cutting image basing on mask size
    result = cutTopAndBottom(coloredMaskedRoad, int(height / 2 - 15), int(height - 130))

    return result