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


# Setting up a Keras model of: 4 Conv and Pool + Flat + 5 Dense
def setupNvidiaModel(inputShape):

    model = Sequential()

    # Convolution Layer 1
    convLayer = Conv2D(filters=12,
                       kernel_size=(5, 5),
                       strides=(1, 1),
                       activation='elu',
                       input_shape=inputShape)
    model.add(convLayer)
    # Pooling Layer 1
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 2
    convLayer = Conv2D(filters=24,
                       kernel_size=(5, 5),
                       activation='elu')
    model.add(convLayer)
    # Pooling Layer 2
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 3
    convLayer = Conv2D(filters=36,
                       kernel_size=(3, 3),
                       activation='elu')
    model.add(convLayer)
    # Pooling Layer 3
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Convolution Layer 4
    convLayer = Conv2D(filters=48,
                       kernel_size=(3, 3),
                       activation='elu')
    model.add(convLayer)
    # Pooling Layer 4
    poolingLayer = MaxPooling2D(pool_size=(2, 2),
                                strides=(2, 2))
    model.add(poolingLayer)

    # Flatten
    model.add(Flatten())

    # Dense layer 1
    denseLayer = Dense(1164,
                       activation='elu')
    model.add(denseLayer)
    # Dense layer 2
    denseLayer = Dense(100,
                       activation='elu')
    model.add(denseLayer)
    # Dense layer 3
    denseLayer = Dense(50,
                       activation='elu')
    model.add(denseLayer)
    # Dense layer 4
    denseLayer = Dense(10,
                       activation='elu')
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
batchSize = 500
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

# Preparing for validation data retrieval
validationSize = int(videoLengthInFrames * 0.15)
validationGap = int(videoLengthInFrames/validationSize)

# Iterating through all couples of frames of the video
coupleCounter = 0
frameCoupleArray = []
frameCounter = 0
batchFrames = []
batchSpeeds = []
evalFrames = []
evalSpeeds = []
flow_mat = None
image_scale = 0.5
nb_images = 1
win_size = 15
nb_iterations = 2
deg_expansion = 5
STD = 1.3
while(coupleCounter < videoLengthInFrames-1):

    # Read a couple of new frames from the video feed
    ret2, newFrame = videoFeed.read()

    # Elaborating image
    newFrameROI = elaborateImage(newFrame)

    # Calculating the optical flow
    if coupleCounter == 0:
        # If this is the first frame...
        oldFrameROI = newFrameROI
        oldFrame = newFrame
        flow = cv2.calcOpticalFlowFarneback(oldFrameROI, newFrameROI,
                                            flow_mat,
                                            image_scale,
                                            nb_images,
                                            win_size,
                                            nb_iterations,
                                            deg_expansion,
                                            STD,
                                            0)
        #flow = opticalFlowDense(oldFrame, newFrame)
        # Also, set up the CNN model
        flowShape = flow.shape
        model = setupNvidiaModel(flowShape)
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
        #flow = opticalFlowDense(oldFrame, newFrame)

    # Check if this frame is for training or validation
    if frameCounter == validationGap:
        frameCounter = 0
        evalFrames.append(flow)
        evalSpeeds.append(speedTruthArray[coupleCounter])
    else:
        # Saving the couple of data and label for training
        batchFrames.append(flow)
        batchSpeeds.append(speedTruthArray[coupleCounter])

    # Incrementing couples counter and swapping frames
    oldFrameROI = newFrameROI
    oldFrame = newFrame
    coupleCounter = coupleCounter + 1
    frameCounter = frameCounter + 1
    cv2.imshow('frame',draw_flow(newFrameROI,flow))
    cv2.waitKey(1)

    print(str(coupleCounter))


# Shuffling data before training
# For training
print("\n\n\n###############################\nSHUFFLING MODEL\n")
unified = list(zip(batchFrames, batchSpeeds))
np.random.shuffle(unified)
batchFrames, batchSpeeds = zip(*unified)
# For validation
unified = list(zip(evalFrames, evalSpeeds))
np.random.shuffle(unified)
evalFrames, evalSpeeds = zip(*unified)


# Training model
print("\n\n\n###############################\nTRAINING MODEL\n")
index = 0
trainBatchFrame = []
trainBatchSpeed = []
frameCounter = 0
while(index < len(batchSpeeds)):
    # Forming batch
    trainBatchFrame.append(batchFrames[index])
    trainBatchSpeed.append(batchSpeeds[index])
    # Training batch
    index = index + 1
    frameCounter = frameCounter + 1
    if frameCounter == batchSize or index==(len(batchSpeeds)-1) :
        print("\nWe are at " + str(index) + "\n")
        # Preparing data
        X = np.array(trainBatchFrame)
        Y = np.array(trainBatchSpeed)
        #with tf.device('/cpu:0'):   #Uncomment to use CPU instead of GPU
        model.fit(x=X,
                  y=Y,
                  verbose=1,
                  epochs=15,
                  batch_size=32,
                  shuffle=True
                  )
        # Resetting counter and x and y arrays
        frameCounter = 0
        trainBatchFrame = []
        trainBatchSpeed = []


# Saving the trained model
model.save('speed_model.h5')  # creates a HDF5 file 'speed_model.h5'


# Evaluation of the model
print("\n\n\n#########################################\nEVALUATION OF THE MODEL\n")
X = np.array(evalFrames)
Y = np.array(evalSpeeds)
scores = model.evaluate(X, Y, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(str(scores))

videoFeed.release()
cv2.destroyAllWindows()




