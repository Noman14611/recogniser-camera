import streamlit as st
import cv2
import face_recognition
import pickle
import pandas as pd
from datetime import datetime

# Load known face encodings
with open("face_encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

st.title("🎯 Face Attendance Tracker")
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

attendance_file = "attendance.csv"

# Load or create attendance file
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(attendance_file, index=False)

attendance_df = pd.read_csv(attendance_file)

# Store already marked names
marked_names = set(attendance_df["Name"])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera error")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

            if name not in marked_names:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_entry = pd.DataFrame([[name, now]], columns=["Name", "Time"])
                new_entry.to_csv(attendance_file, mode='a', header=False, index=False)
                marked_names.add(name)

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()

