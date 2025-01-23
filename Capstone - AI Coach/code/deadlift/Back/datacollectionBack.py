import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import csv
import os
import sys
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)
from squat_landmarks import landmarks

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture("deadlift_back.mov")
EXPORT_PATH = "deadliftBack.csv"
MODEL_PATH = "deadliftBack.pkl"

#write header lables for class and landmark coords
with open(EXPORT_PATH, mode = 'w',newline='') as file:
    writer = csv.writer(file,delimiter=',')
    writer.writerow(landmarks)


def writeToCSV(results, label):
    try:
        #Retrieve 2D array of x, y, z, and visibility values for each landmark
        #Flatten to convert it to 1D
        keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()

        #Write the keypoints into a CSV file
        with open(EXPORT_PATH, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow([label] + keypoints.tolist())
            print("data has been written to csv")

    except Exception as e:
        print(e)
        pass

counter = 0
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

            key = cv2.waitKey(1)
            if (key == ord('1')):    
                writeToCSV(results,"bent")
            elif (key == ord('2')):
                writeToCSV(results,"normal")
            elif (key == ord('0')):
                writeToCSV(results,"standing")
            

        except:
            pass

        #Body Drawings
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)

        cv2.imshow('Webcam Footage',image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

