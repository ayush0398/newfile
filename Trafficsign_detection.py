"BEST MODEL: Road Sign and Lane Detection"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
import imutils
import argparse
import os
import math
import threading
#import pyttsx3
import time


# Lane Departure Warning System
indiacator = False

def roi(image, vertices):
    mask = np.zeros_like(image)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    cropped_img = cv2.bitwise_and(image, mask)
    return cropped_img

def calculate_lane_distance_ratio(lines):
    # Assuming lines are sorted from left to right
    if len(lines) >= 2:
        x1, _, _, _ = lines[0][0]
        _, _, x2, _ = lines[-1][0]
        if x2 - x1 != 0:
            return (x2 - x1) / len(lines)
    return None

def draw_lines(image, hough_lines, width, height, warning_threshold):
    if hough_lines is not None and len(hough_lines) > 0:  # Check if any lines were detected
        # Separate left and right lane lines
        left_lines, right_lines = [], []
        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
            if (y2 - y1) / (x2 - x1) < 0:  # Check the slope to determine left/right
                left_lines.append(line)
                color = (0, 0, 255)  # Red for left line
            else:
                right_lines.append(line)
                color = (0, 255, 0)  # Green for right line

            cv2.line(image, (x1, y1), (x2, y2), color, 2)

        # Calculate lane distance ratio
        lane_distance_ratio = calculate_lane_distance_ratio(left_lines + right_lines)
        print(lane_distance_ratio, "lane_distance_ratio")

        if lane_distance_ratio is not None:
            # Update the warning_threshold dynamically based on the lane distance ratio
            warning_threshold = max(0.1, min(lane_distance_ratio, 0.4))
            print("Updated warning_threshold:", warning_threshold)

        return image, lane_distance_ratio
    else:
        print("No lines detected")
        return image, None



def process_lane_detection(img, consecutive_frames, warning_threshold):
    height, width, _ = img.shape
    
    # Define region of interest (ROI) for lane detection"""
    
    roi_vertices = [
        (0, 850),
        (2 * width / 3, 2 * height / 3),
        (width, height)
    ]
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply dilation to enhance lane markings
    gray_img = cv2.dilate(gray_img, kernel=np.ones((3, 3), np.uint8))

    # Apply Canny edge detection
    canny = cv2.Canny(gray_img, 130, 220)

    # Mask out region outside ROI
    roi_img = roi(canny, np.array([roi_vertices], np.int32))

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(roi_img, 1, np.pi / 180, threshold=10, minLineLength=15, maxLineGap=2.5)

    # Draw detected lines and calculate lane distance ratio
    img_with_lines, lane_distance_ratio = draw_lines(img.copy(), lines, width, height, warning_threshold)

    # Update list of consecutive frames
    consecutive_frames.append(lane_distance_ratio)
    if len(consecutive_frames) > 10:
        consecutive_frames.pop(0)  # Keep only the last 10 frames

    # Check if lane departure warning should be issued
    if all(frame is not None and frame < warning_threshold for frame in consecutive_frames):
        cv2.putText(img_with_lines, "Warning: Lane Departure", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("Warning: Lane Departure")

    return img_with_lines

# Traffic Sign Detection
SIZE = 32

def load_model(filename):
    model = cv2.ml.SVM_load(filename)
    return model

def load_labels(filename):
    with open(filename, 'r') as file:
        labels = file.readlines()
    return labels

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SIZE * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE, SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
def get_hog():
    winSize = (20, 20)
    blockSize = (10, 10)
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)

    return hog

def getLabel(model, data):
    gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    img = [cv2.resize(gray,(SIZE,SIZE))]
    img_deskewed = list(map(deskew, img))
    hog = get_hog()
    hog_descriptors = np.array([hog.compute(img_deskewed[0])])
    hog_descriptors = np.reshape(hog_descriptors, [-1, hog_descriptors.shape[1]])
    return int(model.predict(hog_descriptors)[1][0][0])

def run_speech(speech, speech_message):
    speech.say(speech_message)
    speech.runAndWait()

#def play_sound_for_sign(speech, sign_name, cooldown_duration, last_detection_time):
#    current_time = time.time()
#    if current_time - last_detection_time > cooldown_duration:
#        message = f"Detected sign: {sign_name}"
#        p = threading.Thread(target=run_speech, args=(speech, message))
#        p.start()
#        last_detection_time = current_time  # Update the last detection time
#    return last_detection_time

#speech = pyttsx3.init()


SIGNS = ["ERROR",
        "STOP",
        "TURN LEFT",
        "Stay Left",
        "Bump Ahead",
        "Speed Limit 50",
        "ONE WAY",
        "SPEED LIMIT",
        "OTHER"]



### Preprocess image

def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels=list(channels)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)           # paramter 
    gray = cv2.cvtColor( LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian( gray, cv2.CV_8U,3,3,2)       # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image
    
def binarization(image):
    #thresh = cv2.threshold(image,32,255,cv2.THRESH_BINARY)[1]
    thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return thresh

def preprocess_image(image):
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    return image

# Find Signs
def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def findContour(image):
    #find contours in the thresholded image
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    return cnts[0]

def contourIsSign(perimeter, centroid, threshold):
    #  perimeter, centroid, threshold
    # # Compute signature of contour
    result=[]
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result ]
    # Check signature of contour.
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold: # is  the sign
        return True, max_value + 2
    else:                 # is not the sign
        return False, max_value + 2

#crop sign 
def cropContour(image, center, max_distance):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(center[0] - max_distance), 0])
    bottom = min([int(center[0] + max_distance + 1), height-1])
    left = max([int(center[1] - max_distance), 0])
    right = min([int(center[1] + max_distance+1), width-1])
    print(left, right, top, bottom)
    return image[left:right, top:bottom]

def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height-1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width-1])
    print(top,left,bottom,right)
    return image[top:bottom,left:right]


def findLargestSign(image, contours, threshold, distance_theshold):
    max_distance = 0
    coordinate = None
    sign = None
    for c in contours:
        
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and distance > max_distance and distance > distance_theshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1,2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinate = [(left-2,top-2),(right+3,bottom+1)]
            sign = cropSign(image,coordinate)
    return sign, coordinate


def findSigns(image, contours, threshold, distance_theshold):
    signs = []
    coordinates = []
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, max_distance = contourIsSign(c, [cX, cY], 1-threshold)
        if is_sign and max_distance > distance_theshold:
            sign = cropContour(image, [cX, cY], max_distance)
            signs.append(sign)
            coordinate = np.reshape(c, [-1,2])
            top, left = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis = 0)
            coordinates.append([(top-2,left-2),(right+1,bottom+1)])
    return signs, coordinates

def localization(image, min_size_components, similitary_contour_with_circle, model, count, current_sign_type):
    original_image = image.copy()
    binary_image = preprocess_image(image)

    binary_image = removeSmallComponents(binary_image, min_size_components)

    binary_image = cv2.bitwise_and(binary_image,binary_image, mask=remove_other_color(image))

    #binary_image = remove_line(binary_image)

    cv2.imshow('BINARY IMAGE', binary_image)
    contours = findContour(binary_image)
    #signs, coordinates = findSigns(image, contours, similitary_contour_with_circle, 15)
    sign, coordinate = findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
    
    text = ""
    sign_type = -1
    i = 0

    if sign is not None:
        sign_type = getLabel(model, sign)
        sign_type = sign_type if sign_type <= 8 else 8
        text = SIGNS[sign_type]
        #cv2.imwrite(str(count)+'_'+text+'.png', sign)

    if sign_type > 0 and sign_type != current_sign_type:        
        cv2.rectangle(original_image, coordinate[0],coordinate[1], (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(original_image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
    return coordinate, original_image, sign_type, text

def remove_line(img):
    gray = img.copy()
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 5
    maxLineGap = 3
    lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(mask,(x1,y1),(x2,y2),(0,0,0),2)
    return cv2.bitwise_and(img, img, mask=mask)

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV
    lower_blue = np.array([100, 128, 0])
    upper_blue = np.array([215, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Define range of white color in HSV
    lower_white = np.array([0, 0, 128], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    # Threshold the HSV image to get only white colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Define range of black color in HSV
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([170, 150, 50], dtype=np.uint8)
    # Threshold the HSV image to get only black colors
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Combine masks
    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)

    # Exclude dark clouds (modify the lower and upper range for dark clouds)
    lower_dark_cloud = np.array([0, 0, 30], dtype=np.uint8) 
    upper_dark_cloud = np.array([180, 255, 100], dtype=np.uint8)
    dark_cloud_mask = cv2.inRange(hsv, lower_dark_cloud, upper_dark_cloud)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(dark_cloud_mask))

    # Include light gray clouds
    lower_light_gray_cloud = np.array([0, 0, 150], dtype=np.uint8)
    upper_light_gray_cloud = np.array([180, 30, 255], dtype=np.uint8)
    light_gray_cloud_mask = cv2.inRange(hsv, lower_light_gray_cloud, upper_light_gray_cloud)
    mask = cv2.bitwise_or(mask, light_gray_cloud_mask)

    # Exclude regions corresponding to green trees (assuming they are green)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    tree_mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_or(mask, cv2.bitwise_not(tree_mask))

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    return mask


def main(args):
    consecutive_frames=[]
    warning_threshold=0.2

    
    #Training phase
    #model = "D:\\Traffic-Sign-Detection-master\\data_svm.dat"
    model = load_model("./data_svm.dat")
    labels = load_labels('./labels.txt')
    #model = training()
    #vidcap = cv2.VideoCapture(args.file_name)
    vidcap = cv2.VideoCapture(0)
    #vidcap= cv2.VideoCapture(0)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = vidcap.get(3)  # float
    height = vidcap.get(4) # float

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.avi',fourcc, fps , (720,480))
    #out = cv2.VideoWriter('D:\Save\Insurance\output.avi',fourcc, fps , (720,480))

    # initialize the termination criteria for cam shift, indicating
    # a maximum of ten iterations or movement by a least one pixel
    # along with the bounding box of the ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roiBox = None
    roiHist = None

    success = True
    similitary_contour_with_circle = 0.60  # parameter
    count = 0
    current_sign = None
    current_text = ""
    current_size = 0
    sign_count = 0
    coordinates = []
    position = []
    file = open("Output.txt", "w")
    
    cooldown_duration = 7  # Set the cooldown duration in seconds
    last_detection_time = 0  # Initialize the last detection time

    while True:
        success,frame = vidcap.read()
        if not success:
            print("FINISHED")
            break
        width = frame.shape[1]
        height = frame.shape[0]
        #frame = cv2.resize(frame, (640,int(height/(width/640))))
        frame = cv2.resize(frame, (720,480))

        print("Frame:{}".format(count))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        frame_with_lane_detection = process_lane_detection(frame, consecutive_frames, warning_threshold)
        #out.write(frame_with_lane_detection)
        
        # Process traffic sign detection and localization
        coordinate, image, sign_type, text = localization(frame, args.min_size_components, args.similitary_contour_with_circle, model, count, current_sign)
        if coordinate is not None:
            cv2.rectangle(image, coordinate[0],coordinate[1], (255, 255, 255), 1)
        print("Sign:{}".format(sign_type))
        if sign_type > 0 and (not current_sign or sign_type != current_sign):
            current_sign = sign_type
            current_text = text
            top = int(coordinate[0][1]*1.05)
            left = int(coordinate[0][0]*1.05)
            bottom = int(coordinate[1][1]*0.95)
            right = int(coordinate[1][0]*0.95)

            position = [count, sign_type if sign_type <= 8 else 8, coordinate[0][0], coordinate[0][1], coordinate[1][0], coordinate[1][1]]
            cv2.rectangle(image, coordinate[0],coordinate[1], (0, 255, 0), 1)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)

            tl = [left, top]
            br = [right,bottom]
            print(tl, br)
            current_size = math.sqrt(math.pow((tl[0]-br[0]),2) + math.pow((tl[1]-br[1]),2))
            # grab the ROI for the bounding box and convert it
            # to the HSV color space
            roi = frame[tl[1]:br[1], tl[0]:br[0]]
            #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB) 

            # compute a HSV histogram for the ROI and store the
            # bounding box
            roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
            roiBox = (tl[0], tl[1], br[0], br[1])

        elif current_sign:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)

            # apply cam shift to the back projection, convert the
            # points to a bounding box, and then draw them
            (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
            pts = np.int0(cv2.boxPoints(r))
            s = pts.sum(axis = 1)
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            size = math.sqrt(pow((tl[0]-br[0]),2) +pow((tl[1]-br[1]),2))
            print(size)

            if  current_size < 1 or size < 1 or size / current_size > 30 or math.fabs((tl[0]-br[0])/(tl[1]-br[1])) > 2 or math.fabs((tl[0]-br[0])/(tl[1]-br[1])) < 0.5:
                current_sign = None
                print("Stop tracking")
            else:
                current_size = size

            if sign_type > 0:
                top = int(coordinate[0][1])
                left = int(coordinate[0][0])
                bottom = int(coordinate[1][1])
                right = int(coordinate[1][0])

                position = [count, sign_type if sign_type <= 8 else 8, left, top, right, bottom]
                cv2.rectangle(image, coordinate[0],coordinate[1], (0, 255, 0), 1)
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(image,text,(coordinate[0][0], coordinate[0][1] -15), font, 1,(0,0,255),2,cv2.LINE_4)
            elif current_sign:
                position = [count, sign_type if sign_type <= 8 else 8, tl[0], tl[1], br[0], br[1]]
                cv2.rectangle(image, (tl[0], tl[1]),(br[0], br[1]), (0, 255, 0), 1)
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(image,current_text,(tl[0], tl[1] -15), font, 1,(0,0,255),2,cv2.LINE_4)

    
        if current_sign:
            sign_count += 1
            coordinates.append(position)
            #last_detection_time = play_sound_for_sign(speech, current_text, cooldown_duration, last_detection_time)
        combined_frame = cv2.addWeighted(frame_with_lane_detection, 0.5, image, 0.5, 0)

        # Display the combined frame
        cv2.imshow('Result', combined_frame)
        print(width, height)
        
        count = count + 1
        #Write to video
        #out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #combined_frame.release()
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Traffic Sign Detection with Lane Departure Warning System")
    #parser.add_argument('--file_name', default="D:\\Traffic-Sign-Detection-master\\Front\\(3).mp4", help="Video to be analyzed")
    #parser.add_argument('--file_name', default="D:\\Traffic-Sign-Detection-master\\teest.avi", help="Video to be analyzed")
    #parser.add_argument('--file_name', default="D:\\ADAS\\Back\\(21).mp4", help="Video to be analyzed")
    parser.add_argument('--min_size_components', type=int, default=300, help="Min size component to be reserved")
    parser.add_argument('--similitary_contour_with_circle', type=float, default=0.60, help="Similarity to a circle")
    
    

    args = parser.parse_args()
    main(args)
