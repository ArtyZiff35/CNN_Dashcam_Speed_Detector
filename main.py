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
from imageElaboration import *


# TODO ignore lateral movements
# TODO consider only cone of the road
# TODO Evaluation
# TODO Calculate average in test to soften changes in speed


# Setting up a Keras model of: Norm + 4 Conv and Pool + Flat + 5 Dense
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
while(coupleCounter < videoLengthInFrames-20):

    # Read a couple of new frames from the video feed
    ret2, newFrame = videoFeed.read()

    # Elaborating image
    newFrameROI = elaborateImage(newFrame)

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
                  epochs=50,
                  batch_size=50
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




