import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
from plank_landmarks import landmarks

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)
MODEL_PATH = "plank.pkl"
counter = 0
prob = np.array([0,0,0]) 
stage = None


# #Retrieve model from pickle file
with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)

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
            
            #predict plank position
            predicted = model.predict(X)[0]
            prob = model.predict_proba(X)[0]

            print(predicted,prob)   

            if predicted == "high" and prob.max() >= 0.5:
                stage = "high"
            elif predicted == "low" and prob.max() >= 0.5:
                stage = "low"
            else:
                stage = "normal"
        except:
            pass

        #Body Drawings
        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
        cv2.rectangle(image,(0,0),(500,140),(245,117,56),-1)

        #Arms Stage Display
        cv2.putText(image, 'STAGE', (150,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, stage, (150,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        #Arms Probability Display
        cv2.putText(image, 'PROB', (350,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(image,str(prob[np.argmax(prob)]), (350,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow('Webcam Footage',image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

