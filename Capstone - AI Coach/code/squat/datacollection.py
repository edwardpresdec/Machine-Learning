import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import csv
from squat_landmarks import landmarks

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture("squat_vid.mov")
EXPORT_PATH = "squat.csv"
MODEL_PATH = "squat.pkl"

#write header lables for class and landmark coords
with open(EXPORT_PATH, mode = 'w',newline='') as file:
    writer = csv.writer(file,delimiter=',')
    writer.writerow(landmarks)


def writeToCSV(results, knee_label,hip_label,heel_label,knee_angle, hip_angle, heel_angle ):
    try:
        #Retrieve 2D array of x, y, z, and visibility values for each landmark
        #Flatten to convert it to 1D
        keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()

        #Write the keypoints into a CSV file
        with open(EXPORT_PATH, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([knee_angle,hip_angle,heel_angle] + [knee_label,hip_label,heel_label] + keypoints.tolist())
            print("data has been written to csv")
                

        

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

min_knee_angle = float('inf')

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
           
            left_hip = [landmark_list[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmark_list[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmark_list[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmark_list[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmark_list[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmark_list[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            left_shoulder = [landmark_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmark_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_heel = [landmark_list[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmark_list[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            left_foot = [landmark_list[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmark_list[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

            #Calculate angles
            angle_left_knee = int(calculate_angle(left_hip, left_knee, left_ankle))
            angle_left_hip = int(calculate_angle(left_shoulder, left_hip, left_knee))
            angle_left_heel = int(calculate_angle(left_knee,left_heel,left_foot))

            knee_label = 0
            hip_label = 0
            heel_label = 0

            if (angle_left_knee <min_knee_angle):
                min_knee_angle = angle_left_knee
                if(min_knee_angle < 110):
                    if(min_knee_angle < 50):
                        knee_label = 1
                                
                    if(angle_left_hip < 55):
                        hip_label = 1

                    if(angle_left_heel < 70):
                        heel_label = 1
                    
                    writeToCSV(results,knee_label,hip_label,heel_label,angle_left_knee,angle_left_hip,angle_left_heel)
                    min_knee_angle = 10000
            
            
            # Visualize angle
            cv2.putText(image, "Knee angle: " + str(angle_left_knee),(700,100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3, cv2.LINE_AA)
            
            cv2.putText(image, "Hip Angle: " + str(angle_left_hip),(700,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3, cv2.LINE_AA)
            cv2.putText(image, "Heel Angle: " + str(angle_left_heel),(700,300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3, cv2.LINE_AA)
    

        except:
            pass

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

