# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:53:59 2024

@author: AAA529927
"""
"""
Things to FIX
1. Make it so when looking left or right, the iris alarm goes off. = DONE
2. Make it so when one alarm is running dont run another one, bring universal pause on sound but counts should be running. = Temp_Fix
3. Make all of them in function call. = Started
4. Make sure to properly purge the whole code to avoid avg_ear issue.= Error changed to left_IRIS
5. Remove debugging.
 

"""
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import time
import threading
#import pyttsx3
import warnings
warnings.filterwarnings("ignore")

'''
def run_speech(speech, speech_message):
    speech.say(speech_message)
    speech.runAndWait()
'''
#speech = pyttsx3.init()
#Declare EAR threshold value for blink detection;0<=EAR_threshold<=0.4 
EAR_threshold=0.15

MAR_threshold=0.8
atten_lev_min=0
atten_lev_distraction_max = 180
atten_lev_sleep_max = 240
atten_lev_sleep = atten_lev_sleep_max
atten_lev_distraction = atten_lev_distraction_max
atten_lev_threshold_sleep = 200 #for sleep
atten_lev_threshold_distraction = 150 #for distraction
sleep_count = 0
distracted_count = 0
last_increment_time_sleep = time.time()
last_increment_time_sleep = 0
last_increment_time_distracted = time.time()
last_increment_time_distracted = 0
last_increment_time_sleep_drow = time.time()
last_increment_time_sleep_drow = 0
increment_flag_sleep = True  # Flag to control the increment of sleep_count
increment_flag_distracted = True # Flag to control the increment of distracted_count
drow_left = 0
drow_right = 0
dows=0
pause_after_alarm = False
alarm_start_time = 0 
# Calculate eye aspect ratio by calculating L2 norm or Eucleadean distance between the landmark points
pause_after_alarm_iris = False
alarm_start_time_iris = 0
count_f = 0
def eye_aspect_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])

    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    
    return ear


def iris_aspect_ratio(iris):
    p1_minus_p3 = dist.euclidean(iris[0], iris[2]) 
    print(p1_minus_p3)
    p2_minus_p4 = dist.euclidean(iris[1], iris[3])  
    
    return p1_minus_p3, p2_minus_p4

def euclidean_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# set upper and lower bound of attention_level variable to 500 and 0 respectively
def attention_varible_bound_check_ear(atten_lev_distraction):
    if atten_lev_distraction>atten_lev_distraction_max:
        atten_lev_distraction=atten_lev_distraction_max
    if atten_lev_distraction<atten_lev_min:
        atten_lev_distraction=atten_lev_min
    return atten_lev_distraction

def attention_varible_bound_check_eye(atten_lev_sleep):
    if atten_lev_sleep>atten_lev_sleep_max:
        atten_lev_sleep=atten_lev_sleep_max
    if atten_lev_sleep<atten_lev_min:
        atten_lev_sleep=atten_lev_min
    return atten_lev_sleep
#Defining 12 landmarks for both eyes that we shall consider from the collection of 468 facial landmarks
#These 12 are index positions in the landmark list
RIGHT_EYE=[362, 385, 387, 263, 373, 380]
LEFT_EYE=[33, 160, 158, 133, 153, 144]

#Defining 8 Iris Points
RIGHT_IRIS = [474,475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
#Defining 6 landmarks for mouth
MOUTH=[76,72,302,306,315,85]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# capture webcam video feed for processing
cap = cv2.VideoCapture(0)
ear=[]
mar=[]
headpose=[]
al=[]
blink_left = 0
blink_right = 0
while cap.isOpened():
    success, image = cap.read()
    atten_lev_distraction = attention_varible_bound_check_ear(atten_lev_distraction)
    atten_lev_sleep = attention_varible_bound_check_eye(atten_lev_sleep)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    
    # if landmarks are found in the frame by mediapipe
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z]) 

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            
            if y < -15:
                text = "LEFT"
                atten_lev_distraction -= 1.5
                atten_lev_sleep -= 1.5
                hp = 0
            elif y > 13:
                text = "RIGHT"
                atten_lev_distraction -= 1.5
                atten_lev_sleep -= 1.5
                hp = 0
            elif x < -10:
                text = "DOWN"
                atten_lev_distraction -= 1.5
                atten_lev_sleep -= 1.5
                hp = 0
            elif x > 15:
                text = "UP"
                atten_lev_distraction -= 1.5
                atten_lev_sleep -= 1.5
                hp = 0
            else:
                text = "FRONT"
                atten_lev_distraction += 1.25
                atten_lev_sleep += 1.25
                hp = 1

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            
            cv2.line(image, p1, p2, (255, 0, 0), 2)

            # Add the text on the image
            cv2.putText(image, text=text, fontFace=0, org=(10, 20), fontScale=0.5, color=(0, 0, 255))
            cv2.putText(img=image, text=f"pitch: {round(x, 2)}", fontFace=0, org=(500, 40), fontScale=0.5, color=(0, 255, 0))
            cv2.putText(img=image, text=f"yaw: {round(y, 2)}", fontFace=0, org=(500, 30), fontScale=0.5, color=(0, 255, 0))

        #collect the landmarks of all the facial landmarks
        all_landmarks = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        #save the landmarks for the right and left eye based on the index positions previously declared
        right_eye = all_landmarks[RIGHT_EYE]
        left_eye = all_landmarks[LEFT_EYE]
        mouth = all_landmarks[MOUTH]
        point_362 = all_landmarks[362]
        point_474 = all_landmarks[474]
        point_385 = all_landmarks[385]
        point_475 = all_landmarks[475]

        # Calculate the distances
        distance_362_474 = euclidean_distance(point_362, point_474)
        distance_385_475 = euclidean_distance(point_385, point_475)
        
        point_33 = all_landmarks[33]
        point_471 = all_landmarks[471]
        point_160 = all_landmarks[160]
        point_469 = all_landmarks[469]

        # Calculate the distances
        distance_33_471 = euclidean_distance(point_33, point_471)
        distance_160_469 = euclidean_distance(point_160, point_469)

        # Print or use the distances as needed
        print(f"Distance between point 362 and 474: {distance_362_474}")
        print(f"Distance between point 385 and 475: {distance_385_475}")
        print(f"Distance between point 33 and 471: {distance_33_471}")
        print(f"Distance between point 160 and 469: {distance_160_469}")
        
        
       
          #for counting blink in left eye
        if eye_aspect_ratio(left_eye) < 0.15:  # Change this threshold as needed. 
            blink_left += 1
            print(blink_left)
    
        else:
            left_iris = all_landmarks[LEFT_IRIS] 
            # draw an outline over the landmarks to highlight the eyes in the video frame
            cv2.polylines(image, [left_eye], True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.polylines(image, [left_iris], True, (100, 255, 255), 1, cv2.LINE_AA)
        
        if eye_aspect_ratio(right_eye) < 0.15:
            blink_right += 1
            
        else:
            right_iris = all_landmarks[RIGHT_IRIS]
            
            # draw an outline over the landmarks to highlight the eyes in the video frame
            cv2.polylines(image, [right_eye], True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.polylines(image, [right_iris], True, (0, 255, 255), 1, cv2.LINE_AA)
            
            
        cv2.polylines(image, [left_eye], True, (0,255,0), 1, cv2.LINE_AA)
        cv2.polylines(image, [right_eye], True, (0,255,0), 1, cv2.LINE_AA) 
        cv2.polylines(image, [mouth], True, (0,255,0), 1, cv2.LINE_AA) 
        #Calculate EAR of right eye
        right_EAR=eye_aspect_ratio(right_eye)
        #Calculate EAR of left eye
        left_EAR=eye_aspect_ratio(left_eye)
        #Calculate average EAR with both eyes since we blink simultaneously with both eyes
        avg_EAR=(right_EAR+left_EAR)/2
        mouth_MAR=eye_aspect_ratio(mouth)
        left_Iris=iris_aspect_ratio(left_iris)
        right_Iris = iris_aspect_ratio(right_iris)
        
    # For Distraction
        if avg_EAR<0.15:
            atten_lev_sleep=atten_lev_sleep-1.75
            #cv2.putText(img=image, text="EYES CLOSED", fontFace=0, org=(250, 70), fontScale=0.5, color=(0, 0, 255))
        else:
            atten_lev_sleep=atten_lev_sleep+1.25
        #Calculate Mouth MAR
        
        if mouth_MAR>MAR_threshold:
            atten_lev_sleep=atten_lev_sleep-0.25
        else:
            atten_lev_sleep=atten_lev_sleep+0.125
        
    #For Sleep 
    #    if avg_EAR<0.15:
    #        atten_lev_distraction=atten_lev_distraction-1.75
            #cv2.putText(img=image, text="EYES CLOSED", fontFace=0, org=(250, 70), fontScale=0.5, color=(0, 0, 255))
    #    else:
    #        atten_lev_distraction=atten_lev_distraction+1.25
        #Calculate Mouth MAR
        
        if mouth_MAR>MAR_threshold:
            atten_lev_distraction=atten_lev_distraction-0.0
        else:
            atten_lev_distraction=atten_lev_distraction+1.275

        #Display EAR values in the video frame
        cv2.putText(img=image, text=f"LEFT EAR: {round(left_EAR, 2)}", fontFace=0, org=(10, 30), fontScale=0.5, color=(0, 255, 0))
        cv2.putText(img=image, text=f"RIGHT EAR: {round(right_EAR, 2)}", fontFace=0, org=(10, 40), fontScale=0.5, color=(0, 255, 0))
        cv2.putText(img=image, text=f"AVG EAR: {round(avg_EAR, 2)}", fontFace=0, org=(10, 50), fontScale=0.5, color=(255, 0, 0))
        cv2.putText(img=image, text=f"MOUTH MAR: {round(mouth_MAR, 2)}", fontFace=0, org=(10, 60), fontScale=0.5, color=(255, 0, 0))
        cv2.putText(img=image, text=f"ATTENTION LEVEL Sleep: {round(atten_lev_sleep, 2)}", fontFace=0, org=(10, 70), fontScale=0.5, color=(255, 0, 0))
        cv2.putText(img=image, text=f"ATTENTION LEVEL Distraction: {round(atten_lev_distraction, 2)}", fontFace=0, org=(10, 90), fontScale=0.5, color=(255, 0, 0))
        #print("Angle x",x)
        #print("Angle y",y)
        #print("Angle z",z)
    #this is for IRIS Focus
        if -11 < y < 20:
            if (distance_362_474 > 18.2 and distance_385_475 > 5.8) or (distance_362_474 > 11.6 and distance_385_475 < 1.9) or (distance_33_471 < 3.8 and distance_160_469 < 5.78) or (distance_33_471 > 5.7 and distance_160_469 > 8.8) :
                print("Diastancee",distance_362_474)
                print("Diastanwe",distance_385_475)
                if not pause_after_alarm:
                    count_f += 1
                    print("WW", count_f)
                    if count_f == 90:
                        cv2.putText(img=image, text="ALERT", fontFace=0, org=(250, 300), fontScale=2, color=(0, 0, 255),
                                    thickness=2)
                        print("Its Running")
                        message = 'Hey driver, you are not focusing'
                        p = threading.Thread(target=run_speech, args=(speech, message))
                        p.start()
                        count_f = 0

                if pause_after_alarm and time.time() - alarm_start_time > 10:  # 10 seconds pause after alarm
                    pause_after_alarm = False
            
            else:
                pass
   #make this a function     
   #This is for drowsy threshold
        if blink_left == 7:
            drow_left += 1
            print("D", drow_left)
            blink_left = 0

        if blink_right == 7:
            drow_right += 1
            print("T", drow_right)
            blink_right = 0
            
        if drow_left >= 2 and drow_right >= 2:
            if not pause_after_alarm:
                dows += 1
                last_increment_time_sleep_drow = time.time()
                print("W", dows)

                if dows == 2:
                    cv2.putText(img=image, text="ALERT", fontFace=0, org=(250, 300), fontScale=2, color=(0, 0, 255), thickness=2)
                    print("Its Running")
                    #message = 'Hey driver, you are looking drowsy, wake up wake up wake up'
                    #p = threading.Thread(target=run_speech, args=(speech, message))
                    #p.start()

                if dows == 3:
                    cv2.putText(img=image, text="ALERT", fontFace=0, org=(250, 300), fontScale=2, color=(0, 0, 255), thickness=2)
                    #message = 'Hey driver, you are looking Sleepy, wake up wake up wake up'
                    #p = threading.Thread(target=run_speech, args=(speech, message))
                    #p.start()

                if dows == 4:
                    cv2.putText(img=image, text="ALERT", fontFace=0, org=(250, 300), fontScale=2, color=(0, 0, 255), thickness=2)
                    #message = 'Hey driver, you are looking Sleepy, please park vehicle on the side for your safety'
                    #p = threading.Thread(target=run_speech, args=(speech, message))
                    #p.start()
                    
                if dows == 5:
                    cv2.putText(img=image, text="ALERT", fontFace=0, org=(250, 300), fontScale=2, color=(0, 0, 255), thickness=2)
                    #message = 'Hey driver, you are looking Sleepy, its advised to park vehicle on the side for your safety'
                    #p = threading.Thread(target=run_speech, args=(speech, message))
                    #p.start()

                # Set the flag to pause after alarm
                pause_after_alarm = True

                # Capture the time when the alarm was triggered
                alarm_start_time = time.time()
                if dows > 15:
                    dows = 0
                
        # Check if it's time to resume monitoring
        if pause_after_alarm and time.time() - alarm_start_time > 10:  # 10 seconds pause after alarm
            pause_after_alarm = False
            drow_left = 0
            print("Z",drow_left)
            drow_right = 0
           
        if time.time() - last_increment_time_sleep_drow > 30:  # 15 minutes in seconds
            dows = 0
            print("w",dows)
            
        #make this a function     
        #This is for sleep and distraction
        
        if atten_lev_sleep < atten_lev_threshold_sleep:
            if increment_flag_sleep:
                sleep_count += 1
                #print(sleep_count)

                # Pause the increment for 5 seconds
                increment_flag_sleep = False
                last_increment_time_sleep = time.time()

            if time.time() - last_increment_time_sleep > 5:
                increment_flag_sleep = True

            if sleep_count == 2:
                cv2.putText(img=image, text="ALERT", fontFace=0, org=(250, 300), fontScale=2, color=(0, 0, 255), thickness = 2)
                #message = 'Hey driver, you are looking drowsy, wake up wake up wake up'
                #p = threading.Thread(target=run_speech, args=(speech, message))
                #p.start()

            if sleep_count == 3:
                cv2.putText(img=image, text="ALERT", fontFace=0, org=(250, 300), fontScale=2, color=(0, 0, 255), thickness = 2)
                #message = 'Hey driver, you are looking Sleepy, wake up wake up wake up'
                #p = threading.Thread(target=run_speech, args=(speech, message))
                #p.start()

            if sleep_count == 4:
                cv2.putText(img=image, text="ALERT", fontFace=0, org=(250, 300), fontScale=2, color=(0, 0, 255), thickness = 2)
                #message = 'Hey driver, you are looking Sleepy, please park vehicle on the side for your safety'
                #p = threading.Thread(target=run_speech, args=(speech, message))
                #p.start()
            
                
        if time.time() - last_increment_time_sleep > 30:  # 30 seconds
            sleep_count = 0

        if atten_lev_distraction < atten_lev_threshold_distraction:
            if increment_flag_distracted:
                distracted_count += 1
                print(distracted_count)

                # Pause the increment for 15 seconds
                increment_flag_distracted = False
                last_increment_time_distracted = time.time()

            elif time.time() - last_increment_time_distracted > 5:
                increment_flag_distracted = True

            elif distracted_count == 2:
                cv2.putText(img=image, text="ALERT", fontFace=0, org=(250, 300), fontScale=2, color=(0, 0, 255), thickness = 2)
                #message = 'Hey driver, you are looking distracted, focus focus focus'
                #p = threading.Thread(target=run_speech, args=(speech, message))
                #p.start()

            elif distracted_count == 3:
                cv2.putText(img=image, text="ALERT", fontFace=0, org=(250, 300), fontScale=2, color=(0, 0, 255), thickness = 2)
                #message = 'Hey driver, you are getting distracted, please focus on driving'
                #p = threading.Thread(target=run_speech, args=(speech, message))
                #p.start()

            elif distracted_count == 4:
                cv2.putText(img=image, text="ALERT", fontFace=0, org=(250, 300), fontScale=2, color=(0, 0, 255), thickness = 2)
                #message = 'Hey driver, you are looking distracted, park the vehicle and complete other task first'
                #p = threading.Thread(target=run_speech, args=(speech, message))
                #p.start()

        # Check if it's time to reset distracted_count
        if time.time() - last_increment_time_distracted > 30:  # 15 minutes in seconds
            distracted_count = 0
            

    cv2.imshow('Attention Level Estimation', image)
    al.append(atten_lev_distraction)
    al.append(atten_lev_sleep)
    ear.append(avg_EAR)
    mar.append(mouth_MAR)
    headpose.append(hp)
    # press esc to exit
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()    
    
    
plt.figure(figsize=(12, 8))
plt.tight_layout(pad=5.0)
a1 = plt.subplot(8, 1, 1)
a1.title.set_text("Attention Level")
plt.plot(al)
a2 = plt.subplot(8, 1, 3)    
a2.title.set_text("Eye Aspect Ratio")
plt.plot(ear)
a3 = plt.subplot(8, 1, 5)
a3.title.set_text("Mouth Aspect Ratio")
plt.plot(mar)
a4 = plt.subplot(8, 1, 7)
a4.title.set_text("Headpose")
plt.plot(headpose)
