import cv2
import numpy as np 
import matplotlib.pyplot as plt
import math

#from lane_detection_image import make_coordinates, avg_slope_intercept, canny_edge_detection, display_lines, region_of_interest


def make_coordinates(img, line_parameters):
    global global_x1
    global global_x2
    global global_y1
    global global_y2

    # Retrieve slope and intercept
    #slope, intercept = line_parameters
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0,0

    # Store the height of the image, and set the first y-coordinate to the height,

    if abs(slope) < 0.4:  # Adjust the threshold as needed
        # Handle the case where the slope is close to zero (horizontal line)
        x1 = global_x1
        x2 = global_x2 # Set both x-coordinates to the intercept
        y1 = global_y1  # Set y1 to the bottom of the image
        y2 = global_y2 # Set y2 to 1/5 of the image height

    else:
        # Calculate x-coordinates using slope-intercept formula


        y1 = img.shape[0]
        y2 = int(y1 * (1/5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        global_x1 = x1
        global_x2 = x2
        global_y1 = y1
        global_y2 = y2

    # which means it will start at the bottom of the image.
    #y1 = img.shape[0]

    # Set the second y-coordinate 3/5 of the way up the y-axis of the image.
    #y2 = int(y1 * (1/5))

    # Calculate x-coordinates using slope-intercept formula, y = mx + b, rearranged to x = (y - b) / m
    #x1 = int((y1 - intercept)/slope)
    #x2 = int((y2 - intercept)/slope)

    # Return coordinates in a Numpy array
    
    return np.array([x1, y1, x2, y2])


def avg_slope_intercept(img, lines):
    # left fit list will contain coordinates of the averaged lines on the left and 
    # right fit list will contain the coordinates of the averages lines on the right.
    
    if lines is None:
        return None        
    left_fit = []    
    right_fit = []
    left_line = None
    right_line = None

    ''' Each is a 2-D array containing our line coordinates in the form [[x1, y1, x2, y2]].
        These coordinates specify the line's parameters, as well as the location of the lines w/ respect to the image space, 
        ensuring that they are placed in the correct position.'''

    ''' Loop through lines. Array will need to be reshaped from a 2-D array, in this case, to a 1-D array. 

        *** Note on the Numpy "reshape" function. ***
        If an integer is given as the "newshape" argument for the Numpy reshape function, then the result will be a 1-D array of that length.  
        So in the case below, it will reshape the 2-D array containing our 4 coordinates to a 1-D array with our 4 coordinates. 
        Then we can extract each element to a separate variable.'''
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        # The polyfit function will fit a first degree polynomial (like y = mx + b) to our given coordinates. 
        # In other words, it will return the slope, and y-intercept respectively.
        
        parameters = np.polyfit((x1, x2), (y1, y2), 1) 
        slope = parameters[0]
        intercept = parameters[1]

        ''' If slope is negative then the line is on the left, if slope is positive then the line is on the right.
            *** Note: The y-axis is reversed from the normal way we are used to seeing Cartesian(x,y) coordinates.  Here they decrease as they go up the axis *** ''' 
        if slope < 0:
            left_fit.append((slope, intercept))        
        elif slope > 0:
            right_fit.append((slope, intercept))
    
    # Calculate the average slope and y-intercept for left and right lines to produce one solid line
    try:
        if left_fit:
            left_fit_avg = np.average(left_fit, axis=0)
            left_line = make_coordinates(img, left_fit_avg)
        
        if right_fit:
            right_fit_avg = np.average(right_fit, axis=0)
            right_line = make_coordinates(img, right_fit_avg)

        # Calculate x and y coordinates for left and right line, the return left and right line as a Numpy array
        if left_line is not None and right_line is None:
            return np.array([left_line])
        elif left_line is None and right_line is not None:
            return np.array([right_line])
        elif left_line is not None and right_line is not None:
            if left_line[0] < right_line[2] and right_line[0] < left_line[2]:
                return None
            return np.array([left_line, right_line])
        else:
            return None
    
    except Exception as e:
        print(e)
        return None

def canny_edge_detection(img):
    # Use the cvtColor() function to grayscale the image 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    erosion_kernel = np.ones((3, 3), np.uint8)
    dilation_kernel = np.ones((3, 3), np.uint8)

    # Use cv2 Erosion and Dilation to reduce noise
    eroded = cv2.erode(img_gray, erosion_kernel, iterations=5)
    dilated = cv2.dilate(eroded, dilation_kernel, iterations=5)

    # Apply Gaussian blur to grayscale image for smoothing
    img_blur = cv2.GaussianBlur(dilated, (3,3), 0)

    # Run Canny Edge detector on blurred grayscale image to detect edges
    canny_img = cv2.Canny(img_blur, 250, 350)

    return canny_img

def display_lines(img, lines):

    ''' Create a black image (pixels with all 0's) with the same dimensions as our image.  
        Lines will displayed on top of this black image '''
    
    line_img = np.zeros_like(img)
    
    ''' Each is a 2-D array containing our line coordinates in the form [[x1, y1, x2, y2]].
        These coordinates specify the line's parameters, as well as the location of the lines w/ respect to the image space, 
        ensuring that they are placed in the correct position.'''

    ''' First check to see if lines are detected. 
        If lines are detected, loop through lines.  
        Array will need to be reshaped from a 2-D array, in this case, to a 1-D array. 

        *** Note on the Numpy "reshape" function. ***
        If an integer is given as the "newshape" argument for the Numpy reshape function, then the result will be a 1-D array of that length.  
        So in the case below, it will reshape the 2-D array containing our 4 coordinates to a 1-D array with our 4 coordinates. 
        Then we can extract each element to a separate variable.'''
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.astype(int)
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 30) # Use Open CV line function to draw our detect lines onto our black image

    return line_img

def region_of_interest(img):
    # Set height of region of interest
    height = img.shape[0]
    width = img.shape[1] 
 
    # print(height)
    # print(width)
    # Array of polygons to fill triangular region of interest
    polygons = np.array([
 
    #[(200, height), (1100, height), (550, 250)]
    [(0-width, height), (width * 2, height), (width//2, 0)]
       
    ])
 
    # Mask to black out area outside of area of interest
    mask = np.zeros_like(img)
 
 
    # Fill area of interest
    cv2.fillPoly(mask, polygons, 255)
 
    # Use bitwise-and operation to apply mask to image and only display area of interest
    masked_img = cv2.bitwise_and(img, mask)
 
    return masked_img

#Taking values from averaged_lines to create endpoint values in an array
def array_fill(averaged_lines):  
    if averaged_lines is not None: #Skipping if lines aren't found in the frame
        for line in averaged_lines:
            x1, y1, x2, y2 = line
            cell_x1 = min(max(int(x1 / cell_size_x), 0), grid_cols - 1)
            cell_x2 = min(max(int(x2 / cell_size_x), 0), grid_cols - 1)
            cell_y1 = min(max(int(y1 / cell_size_y), 0), grid_rows - 1)
            cell_y2 = min(max(int(y2 / cell_size_y), 0), grid_rows - 1)
            grid_array[cell_y1, cell_x1] = True
            grid_array[cell_y2, cell_x2] = True
        true_values = np.where(grid_array) 
        true_coords = list(zip(true_values[0], true_values[1])) #Saving values where True is found, packaged to send to other team
        #for coord in true_coords:  #Printing the values since the integration never happened, just to show the values are being created
        #    print(coord)

#Checking through individual videos
#cap = cv2.VideoCapture('Videos/final14.mp4')

# Live video stream
cap = cv2.VideoCapture(0)



firstPass = 0
while (cap.isOpened()):
    ret, frame1 = cap.read()
    if not ret:
        break
    
    #frame = cv2.resize(frame1, (960, 480))
    frame = cv2.resize(frame1, (1280, 720))

    
    #Waiting a few frames for the program to begin, possibly not necessary after a few changes were made
    firstPass = firstPass + 1
    #Used to be variable-sized based on the camera, ended up hardcoding after running into bugs when switching cameras. Can adjust when a final camera is selected
    if firstPass < 2:
        grid_cols = 30
        grid_rows = 21
        cell_size_x = math.floor(frame.shape[1] / grid_cols +12) #"Forcing" the size, otherwise generating array out of bounds errors
        cell_size_y = math.floor(frame.shape[0] / grid_rows +12)
        grid_array = np.empty((grid_rows, grid_cols), dtype = object)


    # Find edges with canny edge detection algorithm 
    canny_test_img = canny_edge_detection(frame)

    #Display masked image with only area of interest
    cropped_canny_img = region_of_interest(canny_test_img)

    # Use Hough transform to detect and draw lines
    #lines = cv2.HoughLinesP(canny_test_img, 2, np.pi/180, 100, np.array([]), minLineLength=70, maxLineGap=5)
    #lines = cv2.HoughLinesP(cropped_canny_img, 2, np.pi/180, 100, np.array([]), minLineLength=200, maxLineGap=100)
    lines = cv2.HoughLinesP(cropped_canny_img, rho = 6, theta = np.pi/180, threshold = 200, lines = np.array([]), minLineLength = 150, maxLineGap = 150)
    #lines = cv2.HoughLinesP(canny_test_img, rho = 6, theta = np.pi/180, threshold = 200, lines = np.array([]), minLineLength = 150, maxLineGap = 150)

    averaged_lines = avg_slope_intercept(frame, lines)

    array_fill(averaged_lines)

    #if averaged_lines[0] is not None or averaged_lines[1] is not None:
    #    line_img = display_lines(frame, averaged_lines)
    #    combined_img = cv2.addWeighted(frame, 0.7, line_img, 1.0, 1.0)
    #else:
    #    combined_img = frame

    # Display detected lines on black image
    line_img = display_lines(frame, averaged_lines)

    # Combine image with detected lines with original image to overlay detected lines on actual lanes
    combined_img = cv2.addWeighted(frame, 0.8, line_img, 1.0, 1.0)

    # Display video()
    #canny_test_img = cv2.resize(canny_test_img, (960, 480))
    #line_img = cv2.resize(line_img, (960, 480))
    #combined_img = cv2.resize(combined_img, (960, 480))

    cv2.imshow("Canny Edge Detection", canny_test_img)
    cv2.imshow("Hough Transform", line_img)
    cv2.imshow("Combined", combined_img)
    cv2.waitKey(1)

