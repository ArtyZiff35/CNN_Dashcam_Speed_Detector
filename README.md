# CNN Dashcam Speed Detector 

This project was inspired by comma.ai's Speed Challenge: given their training video (onboard dashcam video of 20400 frames @20fps) with the respective ground truth speeds, the objective is to train a model that will be able to predict the car's speed in another test video.

## Dataset
The dataset is composed by:
* Training video (20400 frames @ 20fps) with respective ground truth speeds
* Test video (10798 frames @ 20fps)

The training video features Highway driving in the first part, and Town driving in the second (with intersections, stop signals, ...). Also, illumination conditions change a lot during the video, going through tunnels and under bridges. 
![Sample video screenshots](https://github.com/ArtyZiff35/CNN_Dashcam_Speed_Detector/blob/master/images/roadsOverview.png)
Speed values are changing a lot too, being high and homogeneous during Highway sections, and slower and intermittent during city sections. 

![Ground truth speeds chart](https://github.com/ArtyZiff35/CNN_Dashcam_Speed_Detector/blob/master/images/groundTruthSpeedChart.PNG) 

All of this makes the training video very heterogeneous in its features.
The test video also shows both city and highway driving.

## Approach
The adopted approach consists of feeding Optical Flows calculated from pairs of frames to a Convolutional Neural Network.

### Image preprocessing
The aim is to modify the image in a way such that the optical flow can be calculated with a great precision. Considering that the original frames are very dark, the first thing to do is to increase illumination and contrast:
![Illumination and Contrast adjustment](https://github.com/ArtyZiff35/CNN_Dashcam_Speed_Detector/blob/master/images/AdjustedIlluminationContrast.PNG) 

While the road lane the car is in and its immediate surroundings are ideally moving at a uniform speed, the other cars, opposing lanes and sky are moving in a way that is not helping us in determining the camera's speed. For this reason, the image is cut in a way to maintain only the road and its shoulders:
![Image cut](https://github.com/ArtyZiff35/CNN_Dashcam_Speed_Detector/blob/master/images/FIlledShape.PNG) 

The current image could already be fed to the Optical Flow algorithm, but in order to highline some features (such as the lane markings) that are often too dark and homogeneous with the road because of the video's quality, we can try to detect the markings themselves. 
Assuming that lane markings can only be either white or yellow, first of all we need to apply some color thresholding to extract only those two: yellow is extracted from the HSV version of the image, while white from the standard RGB to greyscale conversion.
![Color thresholding](https://github.com/ArtyZiff35/CNN_Dashcam_Speed_Detector/blob/master/images/colorThreshold.PNG) 

Then, the Canny Edge Detection algorithm is applied in order to remove too much uniformity in the markings, which might confuse the Optical Flow:
![Canny Edge](https://github.com/ArtyZiff35/CNN_Dashcam_Speed_Detector/blob/master/images/cannyEdge.PNG) 

Again, we cut the image in order to keep only our lane:
![Canny Edge_Masked](https://github.com/ArtyZiff35/CNN_Dashcam_Speed_Detector/blob/master/images/CannyEdgeMasked.PNG) 

Finally, we add the outlined version of the image to our previous result in order to highlight our lane markings:
![Combined](https://github.com/ArtyZiff35/CNN_Dashcam_Speed_Detector/blob/master/images/Combined.PNG) 



