from keras.models import load_model
import numpy as np
import keras
import cv2
from keras.optimizers import Adam
from imageElaboration import *


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

# Loading the Keras trained model
model = load_model('./savedModels/greyMaskSimple_15epochs_32batch_500section.h5')
model.compile(Adam(lr=0.001),
                  loss="mse",
                  metrics=["mse"])

# Opening testing video
videoFeed = cv2.VideoCapture('./sourceData/train.mp4')
videoLengthInFrames = int(videoFeed.get(cv2.CAP_PROP_FRAME_COUNT))
print(videoLengthInFrames)

# Iterating through all couples of frames of the video
coupleCounter = 0
frameToPredict = [ 0 ]
flow_mat = None
image_scale = 0.5
nb_images = 1
win_size = 15
nb_iterations = 2
deg_expansion = 5
STD = 1.3
oldNumericSpeed = None
while(coupleCounter < videoLengthInFrames-20):

    # Read a couple of new frames from the video feed
    ret2, newFrame = videoFeed.read()

    # Elaborating image
    newFrameROI = elaborateImage(newFrame)

    # Calculating the optical flow
    if coupleCounter == 0:
        # If this is the first frame...
        oldFrameROI = newFrameROI
        flow = cv2.calcOpticalFlowFarneback(oldFrameROI, newFrameROI,
                                            flow_mat,
                                            image_scale,
                                            nb_images,
                                            win_size,
                                            nb_iterations,
                                            deg_expansion,
                                            STD,
                                            0)
        # Also, set up the CNN model
        flowShape = flow.shape
    else:
        flow = cv2.calcOpticalFlowFarneback(oldFrameROI, newFrameROI,
                                            flow_mat,
                                            image_scale,
                                            nb_images,
                                            win_size,
                                            nb_iterations,
                                            deg_expansion,
                                            STD,
                                            0)

    # This format is required by how the model was trained through np arrays
    frameToPredict[0] = flow

    # Making speed prediction
    X = np.array(frameToPredict)
    speedPrediction = model.predict(X)
    numericSpeed = round(speedPrediction[0,0], 2)
    # Calculating average from previous frame to smoothen
    if oldNumericSpeed is None:
        oldNumericSpeed = numericSpeed
    numericSpeed = (numericSpeed + oldNumericSpeed) / 2
    numericSpeed = round(numericSpeed, 2)
    oldNumericSpeed = numericSpeed
    speedText = str(numericSpeed)
    print(speedText)


    # Drawing predicted speed text on image
    cv2.putText(newFrame, speedText, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, color=(51, 153, 255),lineType=cv2.LINE_AA)
    # Drawing actual correct speed on image
    cv2.putText(newFrame, str(round(speedTruthArray[coupleCounter],2)), (640-150, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, color=(51, 255, 51),lineType=cv2.LINE_AA)


    # Incrementing couples counter and swapping frames
    coupleCounter = coupleCounter + 1
    oldFrameROI = newFrameROI
    #print(str(coupleCounter))
    cv2.imshow('frame',newFrame)
    cv2.waitKey(1)

