import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedStyle
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
import csv
import time
from squat_landmarks import squat_landmarks
from deadlift_landmarks import deadlift_landmarks
from plank_landmarks import plank_landmarks
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime
from keras.layers import TFSMLayer
from sklearn.preprocessing import StandardScaler
from bicep_landmarks import bicep_landmarks

formatted_date = datetime.now().strftime("%Y-%m-%d")
year = int(formatted_date[:4])
month = int(formatted_date[5:7])
day = int(formatted_date[8:10])

MODEL_PATH_ARMS = "deadliftArms.pkl"
MODEL_PATH_BACK = "deadliftBack.pkl"
MODEL_PATH_STANCE = "deadliftStance.pkl"
MODEL_PATH_PLANK = "plank.pkl"
MODEL_PATH_SCHEDULE = "schedule.keras"
MODEL_PATH_SQUAT_HEEL = "heel5.keras"
MODEL_PATH_SQUAT_HIP = "hip5.keras"
MODEL_PATH_SQUAT_KNEE = "knee5.keras"
MODEL_PATH_BICEP_CURL = "bicep.pkl"
SCHEDULE_CSV = "exercises.csv"


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

arms_stage = None
back_stage = None
stance_stage = None
probArms = np.array([0, 0, 0])
probBack = np.array([0, 0, 0])
probStance = np.array([0, 0, ])
stage = "down"

# Retrieve model from pickle file
with open(MODEL_PATH_ARMS, 'rb') as file:
    modelArms = pickle.load(file)

with open(MODEL_PATH_BICEP_CURL,'rb') as file:
    modelBicep = pickle.load(file)

# Retrieve model from pickle file
with open(MODEL_PATH_BACK, 'rb') as file:
    modelBack = pickle.load(file)

# Retrieve model from pickle file
with open(MODEL_PATH_STANCE, 'rb') as file:
    modelStance = pickle.load(file)

# Retrieve model from pickle file
with open(MODEL_PATH_PLANK, 'rb') as file:
    modelPlank = pickle.load(file)

modelSquatHeel = load_model('heel5.keras')
modelSquatHip = load_model('hip5.keras')
modelSquatKnee = load_model('knee5.keras')

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


class WorkoutPage(tk.Frame):
    def __init__(self, parent, app, name):
        super().__init__(parent)
        self.app = app
        self.name = name

        label = ttk.Label(self, text=f"Welcome to the {name} Page!", font=("Arial", 16))
        label.pack(pady=20)

        camera_button = ttk.Button(self, text="Open Camera", style='TButton', command=lambda name=name: self.open_camera(name))
        camera_button.pack(pady=10)

        back_button = ttk.Button(self, text="Back to Main Menu", style='TButton', command=self.back_to_main_menu)
        back_button.pack(pady=10)

    def open_camera(self, name):
        print("camera opened: " + name)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Initialize variables for the stopwatch
            start_time = None
            countdown = 3
            while cap.isOpened():
                ret, frame = cap.read()

                # recoloring image since frame is in BGR
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # make detections
                results = holistic.process(image)
                image_height, image_width, _ = image.shape
                # convert image back to BGR to process it
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if(name=="Plank" and start_time is None): 
                    start_time = time.time()
                try:
                    landmark_list = results.pose_landmarks.landmark
                    if name == "Squat":
                        left_hip = [landmark_list[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                    landmark_list[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        left_knee = [landmark_list[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                     landmark_list[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        left_ankle = [landmark_list[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                      landmark_list[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        left_shoulder = [landmark_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                         landmark_list[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_heel = [landmark_list[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                                     landmark_list[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                        left_foot = [landmark_list[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                     landmark_list[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

                        # Calculate angles

                        row = np.array([[res.x, res.y, res.z, res.visibility] for res in landmark_list]).flatten().tolist()

                        angle_left_knee = int(calculate_angle(left_hip, left_knee, left_ankle))
                        angle_left_hip = int(calculate_angle(left_shoulder, left_hip, left_knee))
                        angle_left_heel = int(calculate_angle(left_knee, left_heel, left_foot))

                        A = np.array([angle_left_knee])
                        B = np.array([angle_left_hip])
                        C = np.array([angle_left_heel])

                        predictedHeel = modelSquatHeel.predict(C)[0]
                        print(predictedHeel[0])
                        predictedKnee = modelSquatKnee.predict(A)[0]
                        print(predictedKnee[0])
                        predictedHip = modelSquatHip.predict(B)[0]
                        print(predictedHip[0])
                        if predictedHeel[0] >= 0.5:
                            heel_stage = "in range"
                        else:
                            heel_stage = "not in range"

                        if predictedHip[0] >= 0.5:
                            hip_stage = "in range"
                        else:
                            hip_stage = "not in range"

                        if predictedKnee[0] >= 0.4:
                            knee_stage = "in range"
                        else:
                            knee_stage = "not in range"

                        cv2.rectangle(image, (0, 0), (1100, 140), (245, 117, 56), -1)

                        # Arms Stage Display
                        cv2.putText(image, 'HEEL', (850, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, heel_stage, (850, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                    cv2.LINE_AA)

                        # Back Stage Display
                        cv2.putText(image, 'HIP', (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, hip_stage, (450, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # Stance Stage Display
                        cv2.putText(image, 'KNEE', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, knee_stage, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                    cv2.LINE_AA)


                    elif name == "Deadlift":

                        row = np.array([[res.x, res.y, res.z, res.visibility] for res in landmark_list]).flatten().tolist()

                        X = pd.DataFrame([row], columns=deadlift_landmarks[1:])
                        X.head()

                        # #predict arms position
                        predicted_Arms = modelArms.predict(X)[0]
                        probArms = modelArms.predict_proba(X)[0]

                        # #predict back position
                        predicted_Back = modelBack.predict(X)[0]
                        probBack = modelBack.predict_proba(X)[0]

                        # predict legs stance position
                        predicted_Stance = modelStance.predict(X)[0]
                        probStance = modelStance.predict_proba(X)[0]

                        print(predicted_Stance, probStance)

                        print(predicted_Back, probBack)
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

                        cv2.rectangle(image, (0, 0), (1100, 140), (245, 117, 56), -1)

                        # Arms Stage Display
                        cv2.putText(image, 'ARMS', (850, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, arms_stage, (850, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                    cv2.LINE_AA)

                        # Arms Probability Display
                        cv2.putText(image, 'PROB', (1050, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, str(probArms[np.argmax(probArms)]), (1050, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2, cv2.LINE_AA)

                        # Back Stage Display
                        cv2.putText(image, 'BACK', (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, back_stage, (450, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # Back Probability Display
                        cv2.putText(image, 'PROB', (650, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, str(probBack[np.argmax(probBack)]), (650, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2, cv2.LINE_AA)

                        # Stance Stage Display
                        cv2.putText(image, 'STANCE', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, stance_stage, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                    cv2.LINE_AA)

                        # Stance Probability Display
                        cv2.putText(image, 'PROB', (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, str(probStance[np.argmax(probStance)]), (250, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2, cv2.LINE_AA)


                    elif(name == "Plank"):

                        # Calculate elapsed time
                        if start_time is not None:
                            elapsed_time = time.time() - start_time
                            cv2.putText(image, f"Elapsed Time: {elapsed_time:.1f} s", (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                        row = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            
                        X = pd.DataFrame([row],columns = plank_landmarks[1:])
                        
                        #predict plank position
                        predicted = modelPlank.predict(X)[0]
                        prob = modelPlank.predict_proba(X)[0]

                        print(predicted,prob)   

                        if predicted == "high" and prob.max() >= 0.5:
                            stage = "high"
                        elif predicted == "low" and prob.max() >= 0.5:
                            stage = "low"
                        else:
                            stage = "low"

                        #Arms Stage Display
                        cv2.putText(image, 'STAGE', (150,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(image, stage, (150,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

                        #Arms Probability Display
                        cv2.putText(image, 'PROB', (350,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(image,str(prob[np.argmax(prob)]), (350,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                    
                    elif(name == "Bicep Curl"):
                        counter = 0
                        shoulder = [landmark_list[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmark_list[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        elbow = [landmark_list[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmark_list[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        wrist = [landmark_list[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmark_list[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                        angle = int(calculate_angle(shoulder, elbow, wrist))
                        
                        row = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
                        row = np.insert(row,0,angle)
                        X = pd.DataFrame([row],columns = bicep_landmarks[1:])

                        predicted_stage = modelBicep.predict(X)[0]
                        prob = modelBicep.predict_proba(X)[0]
                        print(predicted_stage)
                        print(prob)
                        print(prob.max())
                        if predicted_stage == "down" and prob.max() >= 0.7:
                            stage = "down"
                        elif stage == "down" and predicted_stage == "up" and prob.max() >= 0.7:
                            stage = "up"
                            counter += 1

                        cv2.rectangle(image,(0,0),(800,140),(245,117,56),-1)
        
                        #Rep Display
                        cv2.putText(image, 'REPS', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(image, str(counter), (20,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                        
                        #Stage Display
                        cv2.putText(image, 'STAGE', (250,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(image, stage, (250,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                        #Probability Display
                        cv2.putText(image, 'PROB', (450,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(image, str(prob.max()), (450,120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

                except:
                    pass

                #Body Drawings
                mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
                #Press Q to Quit
                cv2.putText(image, 'PRESS \'Q\' TO QUIT', (750,700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)

                cv2.imshow('Webcam Footage',image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        

    def back_to_main_menu(self):
        self.app.show_frame("Main Menu")

    def back_to_main_menu(self):
        self.app.show_frame("Main Menu")

import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

class RecommendWorkoutPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        label = ttk.Label(self, text="Recommended Workout:", font=("Arial", 16))
        label.pack(pady=20)

        # Add a label for the dropdown and entry
        entry_label = ttk.Label(self, text="Select workout and click 'Set Completed':", font=("Arial", 12))
        entry_label.pack(pady=10)

        # Load data to get exercise options
        file_path = 'exercises.csv'
        file_path2 = 'dailyexercises.csv'
        data = pd.read_csv(file_path)
        self.data = data
        self.exercise_options = data.columns[3:].tolist()

        # Add the dropdown widget
        self.selected_exercise = tk.StringVar()
        self.dropdown = ttk.Combobox(self, textvariable=self.selected_exercise, values=self.exercise_options)
        self.dropdown.pack(pady=10)

        # Add a button to use the entered text
        set_button = ttk.Button(self, text="Set Completed", command=self.set_completed)
        set_button.pack(pady=10)

        scaler = MinMaxScaler()

        new_model = load_model("schedule.keras")

        data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
        data.drop(['year', 'month', 'day'], axis=1, inplace=True)
        data = data.set_index('date')
        n_steps = 5
        new_data = data.tail(n_steps)

        scaler.fit_transform(data)
        new_data_scaled = scaler.transform(new_data)

        # Create a sequence from the new data
        new_sequence = new_data_scaled.reshape((1, n_steps, new_data_scaled.shape[1]))

        # Make the prediction
        new_prediction = new_model.predict(new_sequence)

        # Inverse transform the prediction to original scale
        new_prediction_rescaled = scaler.inverse_transform(new_prediction)

        for i in range(len(new_prediction_rescaled[0])):
            mod = new_prediction_rescaled[0][i]%12
            new_prediction_rescaled[0][i]-=mod
            if new_prediction_rescaled[0][i] < 5:
                new_prediction_rescaled[0][i] = 0

        self.prediction_rescaled = new_prediction_rescaled[0]
        print(type(formatted_date))
        print(type(new_prediction_rescaled[0]))

        filename = 'dailyexercises.csv'
        with open(filename, 'w', newline="") as file:
            csvwriter = csv.writer(file) # 2. create a csvwriter object
            csvwriter.writerow(["date"] + data.columns.tolist()) # 4. write the header
            csvwriter.writerow([formatted_date] + new_prediction_rescaled[0].tolist())
            csvwriter.writerow([formatted_date] + new_prediction_rescaled[0].tolist())
        
        data2 = pd.read_csv(file_path2)
        self.data2 = data2
        self.value_list = data2.values.tolist()

        headers = data.columns
        self.header_list = headers.tolist()
        self.a = "Perform the following in sets of 12 reps:\n"

        for i in range(len(self.prediction_rescaled)):
            self.a += f" {self.header_list[i]}: {int(self.prediction_rescaled[i])}\n"

        self.additional_text = ttk.Label(self, text=self.a, font=("Arial", 12))
        self.additional_text.pack(pady=10)

        back_button = ttk.Button(self, text="Back to Main Menu", style='TButton', command=self.back_to_main_menu)
        back_button.pack(pady=10)

    def set_completed(self):
        selected_exercise = self.selected_exercise.get()
        main_daily = self.value_list[0]
        change_daily = self.value_list[1]
        if selected_exercise:
            exercise_index = self.exercise_options.index(selected_exercise) + 1
            print(exercise_index)
            new_value = max(0, int(change_daily[exercise_index]) - int(main_daily[exercise_index]//3))
            if new_value<=5:
                new_value = 0
            self.value_list[1][exercise_index] = new_value
            df = pd.read_csv('dailyexercises.csv')
            df.loc[1, self.header_list[exercise_index-1]] = new_value
            df.to_csv('dailyexercises.csv', index=False)
            self.update_recommendations()
            

    def update_recommendations(self):
        self.a = "Perform the following in sets of 12 reps:\n"
        for i in range(len(self.value_list[1])-1):
            self.a += f" {self.header_list[i]}: {int(self.value_list[1][i+1])}\n"
        self.additional_text.config(text=self.a)

    def back_to_main_menu(self):
        self.app.show_frame("Main Menu")



class WorkoutApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Workout App")
        self.geometry("500x300")
        style = ThemedStyle(self)
        style.theme_use('breeze')  # You can use different themes

        self.frames = {}

        # Create main menu frame
        self.main_menu_frame = MainMenu(self)
        self.main_menu_frame.grid(row=0, column=0, sticky="nsew")
        self.frames["Main Menu"] = self.main_menu_frame

        # Create frames for different workouts
        workout_names = ["Squat", "Bicep Curl", "Deadlift", "Plank"]
        for name in workout_names:
            frame = WorkoutPage(self, self, name)
            self.frames[name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Create frame for Recommend Workout
        recommend_workout_frame = RecommendWorkoutPage(self, self)
        self.frames["Recommend Workout"] = recommend_workout_frame
        recommend_workout_frame.grid(row=0, column=0, sticky="nsew")

        # Show the main menu frame initially
        self.show_frame("Main Menu")

    def show_frame(self, page_name):
        frame = self.frames.get(page_name)
        if frame:
            frame.tkraise()


class MainMenu(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # Create a new frame with a background color of sky blue
        menu_frame = tk.Frame(self, bg="sky blue", width=700, height=400)
        menu_frame.pack()

        label = ttk.Label(menu_frame, text="Select a Workout:", font=("Arial", 16))
        label.pack(pady=20)

        # Create buttons for each exercise with vertical spacing
        workout_names = ["Squat", "Bicep Curl", "Deadlift", "Plank"]
        for name in workout_names:
            button = ttk.Button(menu_frame, text=name, style='TButton', command=lambda n=name: self.parent.show_frame(n))
            button.pack(pady=10)

        # Create a button for recommending workout
        recommend_button = ttk.Button(menu_frame, text="Recommend Workout", style='TButton',
                                      command=lambda: self.parent.show_frame("Recommend Workout"))
        recommend_button.pack(pady=10)


if __name__ == "__main__":
    app = WorkoutApp()
    app.mainloop()
