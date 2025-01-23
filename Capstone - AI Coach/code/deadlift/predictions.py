import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import csv
from squat_landmarks import landmarks

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
MODEL_PATH_ARMS = "deadliftArms.pkl"
MODEL_PATH_BACK = "deadliftBack.pkl"
MODEL_PATH_STANCE = "deadliftStance.pkl"

counter = 0
arms_stage = None
back_stage = None
stance_stage = None
probArms = np.array([0,0,0]) 
probBack = np.array([0,0,0]) 
probStance = np.array([0,0,]) 
predicted_Back = None
predicted_Arms = None
predicted_Stance = None


# #Retrieve model from pickle file
with open(MODEL_PATH_ARMS,'rb') as file:
    modelArms = pickle.load(file)

#Retrieve model from pickle file
with open(MODEL_PATH_BACK,'rb') as file:
    modelBack = pickle.load(file)

#Retrieve model from pickle file
with open(MODEL_PATH_STANCE,'rb') as file:
    modelStance = pickle.load(file)

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
            
            row = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            
            X = pd.DataFrame([row],columns = landmarks[1:])
            
            # #predict arms position
            predicted_Arms = modelArms.predict(X)[0]
            probArms = modelArms.predict_proba(X)[0]

            # #predict back position
            predicted_Back = modelBack.predict(X)[0]
            probBack = modelBack.predict_proba(X)[0]

            #predict legs stance position
            predicted_Stance = modelStance.predict(X)[0]
            probStance = modelStance.predict_proba(X)[0]

            print(predicted_Stance,probStance)
            
            print(predicted_Back,probBack)
            if predicted_Arms == "wide" and probArms.max() >= 0.6:
                arms_stage = "wide"
            elif predicted_Arms == "narrow" and probArms.max() >= 0.6:
                arms_stage = "narrow"
            else:
                arms_stage = "normal"

            if predicted_Back == "normal" and probBack.max() >= 0.6:
                back_stage = "normal"
            elif predicted_Back == "bent" and probBack.max() >= 0.6:
                back_stage = "bent"
            else:
                back_stage = "standing"

            if predicted_Stance == "wide" and probStance.max() >= 0.6:
                stance_stage = "wide"
            elif predicted_Stance == "narrow" and probStance.max() >= 0.6:
                stance_stage = "narrow"
            else:
                stance_stage = "normal"
        except:
            pass

        #Body Drawings
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
        cv2.rectangle(image,(0,0),(1100,140),(245,117,56),-1)

        #Arms Stage Display
        cv2.putText(image, 'ARMS', (850,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, arms_stage, (850,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        #Arms Probability Display
        cv2.putText(image, 'PROB', (1050,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image,str(probArms[np.argmax(probArms)]), (1050,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        #Back Stage Display
        cv2.putText(image, 'BACK', (450,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, back_stage, (450,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        #Back Probability Display
        cv2.putText(image, 'PROB', (650,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, str(probBack[np.argmax(probBack)]), (650,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        #Stance Stage Display
        cv2.putText(image, 'STANCE', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, stance_stage, (50,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        #Stance Probability Display
        cv2.putText(image, 'PROB', (250,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, str(probStance[np.argmax(probStance)]), (250,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)


        cv2.imshow('Webcam Footage',image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

