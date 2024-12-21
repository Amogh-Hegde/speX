from speech_recognition import listen_and_respond
from face_recognition_util import identify_person
from text_reader import read_text
from datetime import datetime, date
from pyttsx3 import init
from num2words import num2words

import cv2
import face_recognition
import os
import numpy as np

path = "C:\\Users\\Harsh\\OneDrive\\Desktop\\images"
images = []
classNames = [os.path.splitext(cl)[0] for cl in os.listdir(path)]
for cl in os.listdir(path):
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)

encodeListKnown = [face_recognition.face_encodings(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[0] for img in images]

def identify_person(engine):
    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    cap.release()
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    if facesCurFrame:
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            name = classNames[matchIndex].upper() if matches[matchIndex] else "Unknown"
            engine.say(f"{name} is in front of you.")
            engine.runAndWait()
    else:
        engine.say("No person detected.")
        engine.runAndWait()


# Text-to-speech engine initialization
engine = init()
engine.setProperty('rate', 150)

def respond_to_command(command):
    if 'time' in command:
        now = datetime.now()
        hour = int(now.strftime("%I"))
        minute = int(now.strftime("%M"))
        response = f"It is {num2words(hour)} {num2words(minute)} {now.strftime('%p')} now."
        engine.say(response)
        engine.runAndWait()

    elif 'date' in command or "today's" in command:
        today = date.today()
        engine.say(f"Today's date is {today.strftime('%B %d, %Y')}")
        engine.runAndWait()

    elif 'read' in command or 'text' in command:
        read_text(engine)

    elif 'who' in command or 'person' in command:
        identify_person(engine)

if __name__ == "__main__":
    while True:
        command = listen_and_respond()
        if command:
            respond_to_command(command)
