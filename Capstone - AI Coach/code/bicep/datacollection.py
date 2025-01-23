import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import csv
from squat_landmarks import landmarks

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture("majd_vid.mp4")
EXPORT_PATH = "bicep_curl_dataset.csv"
MODEL_PATH = "bicep.pkl"

#write header lables for class and landmark coords
with open(EXPORT_PATH, mode = 'w',newline='') as file:
    writer = csv.writer(file,delimiter=',')
    writer.writerow(landmarks)


def writeToCSV(results,output_label,angle):
    
    try:
        #retreive 2D array of x,y,z, and visibility values for each landmark
        #flatten to convert it to 1D
        keypoints = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        
        # Write the keypoints into a  CSV file
        with open(EXPORT_PATH, mode='a', newline='') as file:
            writer = csv.writer(file,delimiter=',')
            writer.writerow([output_label] + [angle] + keypoints.tolist())

        print("Data has been written to", EXPORT_PATH)

    except Exception as e:
        print(e)
        pass




def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

counter_left = 0
counter_right = 0
counter = 0
stage_left = None
stage_right = None
stage = None

#initiate holistic model
with mp_holistic.Holistic(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        #recoloring image since frame is in BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #make detections
        results = holistic.process(image)
        image_height, image_width, _ = image.shape
        #convert image back to BGR to process it
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)        

        try:
            landmark_list = results.pose_landmarks.landmark
            shoulderLeft = [landmark_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmark_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbowLeft = [landmark_list[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmark_list[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wristLeft = [landmark_list[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmark_list[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            shoulderRight = [landmark_list[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmark_list[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbowRight = [landmark_list[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmark_list[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wristRight = [landmark_list[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmark_list[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Calculate angle
            angleLeft = int(calculate_angle(shoulderLeft, elbowLeft, wristLeft))

            angleRight = int(calculate_angle(shoulderRight, elbowRight, wristRight))
            
            #manually write to CSV file using keys
            key = cv2.waitKey(1)
            if (key == 119):
                writeToCSV(results,"up",angleLeft)
            elif (key == 115):
                writeToCSV(results,"down",angleLeft)
            elif (key == 117):
                writeToCSV(results,"up",angleRight)
            elif (key == 100):
                writeToCSV(results,"down",angleRight)
                
            
            # Visualize angle
            cv2.putText(image, "Left Elbow angle: " + str(angleLeft),(700,100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3, cv2.LINE_AA)
            
            cv2.putText(image, "Right Elbow angle: " + str(angleRight),(700,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3, cv2.LINE_AA)
            
            if (angleLeft > 150):
                stage_left = "down"
            elif (angleLeft < 70) and stage_left =='down':
                stage_left="up"
                counter_left +=1

            if (angleRight > 150):
                stage_right = "down"
            elif(angleRight < 90) and stage_right == "down":
                stage_right = "up"
                counter_right +=1

        except:
            pass


        if stage_left == "up" or stage_right == "up":
            stage = "up"
        elif stage_left == "down" and stage_right == "down":
            stage = "down"

        counter = counter_left+counter_right

        #Body Drawings
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)

        #cv2.rectangle(image,(0,0),(600,140),(245,117,56),-1)
        
        #Rep Display
        # cv2.putText(image, 'REPS', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3, cv2.LINE_AA)
        # cv2.putText(image, str(counter), (20,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        #Stage Display
        # cv2.putText(image, 'STAGE', (250,50), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3, cv2.LINE_AA)
        # cv2.putText(image, stage, 
        #             (250,120), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('Webcam Footage',image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

