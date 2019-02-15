import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model
from tensorflow.python.client import device_lib
from keras import backend as K
import tensorflow as tf
import cv2


# TODO ignore lateral movements
# TODO consider only cone of the road
# TODO Evaluation
# TODO Calculate average in test to soften changes in speed


def drawHoughTransformLines(img, lines):
    if lines is None:
        return img
    a, b, c = lines.shape
    for i in range(a):
        cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 3, cv2.LINE_AA)

    return img


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def thresholdWhiteAndYellow(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_image, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)

    return mask_yw_image





# This method draws the optical flow onto img with a given step (distance between one arrow origin and the other)
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

# This method cuts top and bottom portions of the frame (which are only black areas of the car's dashboard or sky)
def cutTopAndBottom(img):
    height, width = img.shape
    heightBeginning = 20
    heightEnd = height - 30
    crop_img = img[heightBeginning : heightEnd, 0 : width]
    return crop_img

def setupNvidiaModel(inputShape):
    model = Sequential()

    # Normalization layer
    normLayer = keras.layers.BatchNormalization()
    model.add(normLayer)

    # Convolution Layer 1
    convLayer = Conv2D(filters=12,
                       kernel_size=(5, 5),
                       strides=(1, 1),
                       activation='relu',
                       input_shape=inputShape)
    model.add(convLayer)
    # Pooling Layer 1
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 2
    convLayer = Conv2D(filters=24,
                       kernel_size=(5, 5),
                       activation='relu')
    model.add(convLayer)
    # Pooling Layer 2
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 3
    convLayer = Conv2D(filters=36,
                       kernel_size=(3, 3),
                       activation='relu')
    model.add(convLayer)
    # Pooling Layer 3
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 4
    convLayer = Conv2D(filters=48,
                       kernel_size=(3, 3),
                       activation='relu')
    model.add(convLayer)
    # Pooling Layer 4
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Flatten
    model.add(Flatten())

    # Dense layer 1
    denseLayer = Dense(1164,
                       activation='relu')
    model.add(denseLayer)
    # Dense layer 2
    denseLayer = Dense(100,
                       activation='relu')
    model.add(denseLayer)
    # Dense layer 3
    denseLayer = Dense(50,
                       activation='relu')
    model.add(denseLayer)
    # Dense layer 4
    denseLayer = Dense(10,
                       activation='relu')
    model.add(denseLayer)
    # Dense layer 5
    denseLayer = Dense(1)
    model.add(denseLayer)

    # Compilation
    model.compile(Adam(lr=0.001),
                  loss="mse",
                  metrics=["mse"])

    return model


def setupTestModel(inputShape):
    model = Sequential()
    # Adding the first convolutional layer
    convLayer = Conv2D(filters=16,
                       kernel_size=(5, 5),
                       strides=(1, 1),
                       activation='relu',
                       input_shape=inputShape)
    model.add(convLayer)
    # First pooling
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)
    # Adding the second convolutional layer
    convLayer = Conv2D(filters=32,
                       kernel_size=(5, 5),
                       activation='relu')
    model.add(convLayer)
    # Adding the second pooling layer
    poolingLayer = MaxPooling2D(pool_size=(2, 2))
    model.add(poolingLayer)
    # Flatten
    model.add(Flatten())
    # Dense layer 1
    denseLayer = Dense(100,
                       activation='relu')
    model.add(denseLayer)
    # Dense layer 2
    denseLayer = Dense(1)
    model.add(denseLayer)

    # Compilation
    model.compile(Adam(lr=0.001),
                  loss="mse",
                  metrics = ["mse"])

    return model


############ IMPORTANT VARIABLES ###########
batchSize = 200
############################################


# Reading all the speed ground truths
print("Reading speed ground truths")
file = open("./sourceData/train.txt")
speedTruthArrayString = file.readlines()
speedTruthArray = []
for numeric_string in speedTruthArrayString:
    numeric_string = numeric_string.strip('\n')
    speedTruthArray.append(float(numeric_string))
file.close()
print("Read " + str(len(speedTruthArray)) + " values")

# Uncomment to check if GPU is correcetly detected
# print(device_lib.list_local_devices())
# print(K.tensorflow_backend._get_available_gpus())

# GPU settings to allow memory growth
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.log_device_placement=True
# sess = tf.Session(config=config)  #With the two options defined above


# Opening training video
videoFeed = cv2.VideoCapture('./sourceData/train.mp4')
videoLengthInFrames = int(videoFeed.get(cv2.CAP_PROP_FRAME_COUNT))

# Iterating through all couples of frames of the video
coupleCounter = 0
frameCoupleArray = []
frameCounter = 0
batchFrames = []
batchSpeeds = []
while(coupleCounter < videoLengthInFrames-50):

    # Read a couple of new frames from the video feed
    ret2, newFrame = videoFeed.read()

    # Adjusting brightness and contrast
    newFrame = apply_brightness_contrast(newFrame, 100, 100)

    # Threshold so that only yellow and white are kept. Result is greyscale
    newFrameThreshold = thresholdWhiteAndYellow(newFrame)

    # Apply Gaussian blur to reduce noise
    newFrameBlurred = cv2.GaussianBlur(newFrameThreshold, (5,5),0)

    # Applying canny edge detection
    newFrameEdges = cv2.Canny(newFrameBlurred, 100, 200)

    # Cutting a region of interest
    height, width = newFrameEdges.shape
    # Creating white polygonal shape on black image
    bottomLeft = [10, height-110]
    topLeft = [width/3+60, height/2]
    topRight = [width*2/3-60, height/2]
    bottomRight = [width-10, height-110]
    pts = np.array([bottomLeft, topLeft, topRight, bottomRight], np.int32)
    pts = pts.reshape((-1, 1, 2))
    blackImage = np.zeros((height, width, 1), np.uint8)
    polygonalShape = cv2.fillPoly(blackImage, [pts], (255, 255, 255))
    # Doing AND operation with newFrameEdges
    newFrameROI = cv2.bitwise_and(newFrameEdges, newFrameEdges, mask=polygonalShape)

    # Hough transform to detect straight lines. Returns an array of r and theta values
    lines = cv2.HoughLinesP(newFrameROI, 1, np.pi / 180, 15)
    blackImage = np.zeros((height, width, 1), np.uint8)
    linesDrawn = drawHoughTransformLines(blackImage, lines)



    # Calculating the optical flow
    if coupleCounter == 0:
        # If this is the first frame...
        oldFrameROI = newFrameROI
        flow = cv2.calcOpticalFlowFarneback(oldFrameROI, newFrameROI, 0.5, 0.5, 5, 20, 3, 5, 1.2, 0)
        # Also, set up the CNN model
        flowShape = flow.shape
        model = setupNvidiaModel(flowShape)
    else:
        flow = cv2.calcOpticalFlowFarneback(oldFrameROI, newFrameROI, 0.5, 0.5, 5, 20, 3, 5, 1.2, 0)

    # Saving the couple of data and label
    batchFrames.append(flow)
    batchSpeeds.append(speedTruthArray[coupleCounter])

    # Incrementing couples counter and swapping frames
    oldFrameROI = newFrameROI
    coupleCounter = coupleCounter + 1
    #cv2.imshow('frame', draw_flow(cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY), flow))
    #cv2.imshow('frame', newFrameROI)
    cv2.imshow('frame',draw_flow(newFrameROI,flow))
    cv2.waitKey(1)

    print(str(coupleCounter))

    # Training batch
    frameCounter = frameCounter + 1
    if frameCounter == batchSize or (coupleCounter+1) == videoLengthInFrames:
        # Preparing data
        X = np.array(batchFrames)
        Y = np.array(batchSpeeds)
        #with tf.device('/cpu:0'):   #Uncomment to use CPU instead of GPU
        model.fit(x=X,
                  y=Y,
                  verbose=1,
                  epochs=5,
                  batch_size=20
                  )
        # Resetting counter and x and y arrays
        frameCounter = 0
        batchFrames = []
        batchSpeeds = []


# Saving the trained model
model.save('speed_model.h5')  # creates a HDF5 file 'speed_model.h5'
#model = load_model('speed_model.h5')

videoFeed.release()
cv2.destroyAllWindows()




