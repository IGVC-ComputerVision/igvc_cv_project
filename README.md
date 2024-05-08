**Lane Detection**

We have used OpenCV and NumPy to implement a basic lane detection strategy. In our approach, video images received in real-time will be processed by first converting the images to gray-scale, applying a Gaussian blur filter to smooth the edges of the image, Canny edge detection will then be used to identify edges within a predefined region of interest, and Hough transforms will be used to identify the position of the lane lines.  Lane lines are then mapped onto the video image with a mask to show where the lane lines are located in real-time.

**Object Detection**

Yolo v5 was used to train a model to identify traffic cones in order to implement an object detection and object avoidance strategy.



**Link to website:** https://websites.uta.edu/cseseniordesign/2024/04/25/intelligent-ground-vehicle-igvc-computer-vision/

Reference video as our base of our code, modified it heavily to fit the IGVC's needs: https://www.youtube.com/watch?v=eLTLtUVuuy4&t=4531s
