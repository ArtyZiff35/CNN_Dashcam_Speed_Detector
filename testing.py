from keras.models import load_model
import numpy as np
import keras
import cv2


# This method cuts top and bottom portions of the frame (which are only black areas of the car's dashboard or sky)
def cutTopAndBottom(img):
    height, width = img.shape
    heightBeginning = 20
    heightEnd = height - 30
    crop_img = img[heightBeginning : heightEnd, 0 : width]
    return crop_img


# Reading all the speed ground truths
print("Reading speed ground truths")
file = open("./sourceData/test.txt")
speedTruthArrayString = file.readlines()
speedTruthArray = []
for numeric_string in speedTruthArrayString:
    numeric_string = numeric_string.strip('\n')
    speedTruthArray.append(float(numeric_string))
file.close()
print("Read " + str(len(speedTruthArray)) + " values")

# Loading the Keras trained model
model = load_model('speed_model.h5')

# Opening testing video
videoFeed = cv2.VideoCapture('./sourceData/test.mp4')
videoLengthInFrames = int(videoFeed.get(cv2.CAP_PROP_FRAME_COUNT))
print(videoLengthInFrames)

# Reading the first frame
coupleCounter = 0
frameCoupleArray = []
ret1, oldFrame = videoFeed.read()
oldFrameGrey = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)

# Saving the size of the flow
oldFrameGrey = cutTopAndBottom(oldFrameGrey)
oldFrameGrey = cv2.equalizeHist(oldFrameGrey)
dummyFlow = cv2.calcOpticalFlowFarneback(oldFrameGrey , oldFrameGrey, 0.5, 0.5, 5, 20, 3, 5, 1.2, 0)
flowShape = dummyFlow.shape  # Original non cropped size is (480, 640, 2)

# Iterating through all couples of frames of the video
frameToPredict = [ 0 ]
while(coupleCounter < videoLengthInFrames-20):

    # Read a couple of new frames from the video feed
    ret2, newFrame = videoFeed.read()

    # Convert to greyscale
    newFrameGrey = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)

    # Cut top and bottom portions of the image
    newFrameGrey = cutTopAndBottom(newFrameGrey)

    # Apply Histogram Equalization to increase contrast
    newFrameGrey = cv2.equalizeHist(newFrameGrey)

    # Calculate flow for this couple
    flow = cv2.calcOpticalFlowFarneback(oldFrameGrey , newFrameGrey, 0.5, 0.5, 5, 20, 3, 5, 1.2, 0)
    frameToPredict[0] = flow    # This format is required by how the model was trained through np arrays

    # Making speed prediction
    X = np.array(frameToPredict)
    speedPrediction = model.predict(X)
    speedText = str(round(speedPrediction[0,0], 2))

    # Drawing predicted speed text on image
    cv2.putText(newFrame, speedText, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, color=(51, 153, 255),lineType=cv2.LINE_AA)
    # Drawing actual correct speed on image
    cv2.putText(newFrame, str(round(speedTruthArray[coupleCounter],2)), (640-150, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, color=(51, 255, 51),lineType=cv2.LINE_AA)


    # Incrementing couples counter and swapping frames
    coupleCounter = coupleCounter + 1
    oldFrameGrey = newFrameGrey
    #print(str(coupleCounter))
    cv2.imshow('frame',newFrame)
    cv2.waitKey(1)

