# CNN Dashcam Speed Detector 

This project was inspired by comma.ai's Speed Challenge: given their training video (onboard dashcam video of 20400 frames @20fps) with the respective ground truth speeds, the objective is to train a model that will be able to predict the car's speed in another test video.

## Dataset
The dataset is composed by:
* Training video (20400 frames @ 20fps) with respective ground truth speeds
* Test video (10798 frames @ 20fps)

The training video features Highway driving in the first part, and Town driving in the second (with intersections, stop signals, ...). Also, illumination conditions change a lot during the video, going through tunnels and under bridges. 

Speed values are changing a lot too, being high and homogeneous during Highway sections, and slower and intermittent during city sections.  All of this makes the training video very heterogeneous in its features.

The test video also shows both city and highway driving.
