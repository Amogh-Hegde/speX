# src/main/integrated_system.py

import cv2
import time
import threading
from queue import Queue
import speech_recognition as sr
import pyttsx3
import numpy as np
from datetime import datetime
import sys
import os
from pathlib import Path

# Debug prints
print("Current working directory:", os.getcwd())
print("File location:", __file__)

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent
print("Project root:", PROJECT_ROOT)

# Add the project root to Python path
sys.path.append(str(PROJECT_ROOT))
print("Python path:", sys.path)

try:
    # Now import modules
    print("Attempting to import modules...")
    from text_reader import TextReader, TextMode
    from face_recognition_manager import FaceRecognitionManager
    from recognition.gesture_recognizer import GestureRecognizer
    from recognition.object_detector import ObjectDetector
    print("All modules imported successfully!")
except Exception as e:
    print(f"Import error: {str(e)}")
    print(f"Looking for modules in: {PROJECT_ROOT}/recognition/")
    print(f"Files in recognition directory: {os.listdir(PROJECT_ROOT/'recognition')}")


class IntegratedSystem:
    def __init__(self):
        """
        Main system that integrates all components:
        - Face Recognition
        - Gesture Recognition
        - Object Detection
        - Text Reading
        """
        print("Initializing Integrated Assistant System...")
        
        # Initialize voice components
        self.recognizer = sr.Recognizer()
        self.speaker = pyttsx3.init()
        self.speaker.setProperty('rate', 150)
        self.speaker.setProperty('volume', 0.9)
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Could not access camera")
            
        # Initialize recognition systems
        self.face_recognizer = FaceRecognitionManager()
        self.gesture_recognizer = GestureRecognizer()
        self.object_detector = ObjectDetector()
        self.text_reader = TextReader()
        
        # Share camera with all components
        self.share_camera()
        
        # Set up processing queues
        self.command_queue = Queue()
        self.result_queue = Queue()
        
        # System state
        self.is_running = True
        self.last_activity = time.time()
        self.timeout_duration = 300  #5 minutes
        
        print("System initialization complete!")

    def share_camera(self):
        """Share single camera instance with all components"""
        self.face_recognizer.initialize_camera(self.camera)
        self.gesture_recognizer.initialize_camera(self.camera)
        self.object_detector.initialize_camera(self.camera)
        self.text_reader.initialize_camera(self.camera)

    def speak(self, text):
        """Convert text to speech"""
        print(f"Assistant: {text}")
        self.speaker.say(text)
        self.speaker.runAndWait()

    def listen(self):
        """Listen for voice commands"""
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text.lower()
            except sr.WaitTimeoutError:
                return ""
            except sr.UnknownValueError:
                self.speak("I didn't catch that. Could you please repeat?")
                return ""
            except sr.RequestError:
                self.speak("I'm having trouble with speech recognition.")
                return ""

    def continuous_monitoring(self):
        """
        Continuously monitor environment for:
        - Important objects
        - Known faces
        - Gestures
        - Text
        """
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            # Process environment
            objects = self.object_detector.detect_objects(frame)
            faces = self.face_recognizer.process_frame(frame)
            gestures = self.gesture_recognizer.detect_gestures(frame)
            
            # Check for important changes or hazards
            if self.check_important_changes(objects, faces, gestures):
                description = self.generate_environment_description(
                    objects, faces, gestures
                )
                self.speak(description)
            
            # Check for system timeout
            if time.time() - self.last_activity > self.timeout_duration:
                self.speak("No activity detected for a while. Going to sleep mode.")
                self.is_running = False
                break
            
            time.sleep(0.1)  # Prevent excessive CPU usage

    def check_important_changes(self, objects, faces, gestures):
        """Check for significant changes that need immediate attention"""
        # Check for high-priority objects (potential hazards)
        if any(obj['priority'] == 'high' for obj in objects):
            return True
            
        # Check for new faces
        if any(face['name'] == 'Unknown' for face in faces):
            return True
            
        # Check for important gestures
        if any(gesture in ['wave', 'help'] for gesture in gestures):
            return True
            
        return False

    def generate_environment_description(self, objects, faces, gestures):
        """Create comprehensive environment description"""
        descriptions = []
        
        # Describe people
        if faces:
            known_faces = [f for f in faces if f['name'] != 'Unknown']
            unknown_faces = [f for f in faces if f['name'] == 'Unknown']
            
            if known_faces:
                faces_desc = ", ".join(f"{f['name']} ({f['relation']})" for f in known_faces)
                descriptions.append(f"I see {faces_desc}")
            
            if unknown_faces:
                descriptions.append(f"and {len(unknown_faces)} unknown person(s)")
        
        # Describe objects
        if objects:
            obj_desc = self.object_detector.generate_description(objects)
            descriptions.append(obj_desc)
        
        # Describe gestures
        if gestures:
            gesture_desc = self.gesture_recognizer.get_gesture_description(gestures)
            descriptions.append(gesture_desc)
        
        return ". ".join(descriptions)

    def process_command(self, command):
        """Process voice commands"""
        self.last_activity = time.time()
        
        if not command:
            return False
            
        try:
            # Face recognition commands
            if 'who' in command or 'recognize' in command:
                ret, frame = self.camera.read()
                if ret:
                    faces = self.face_recognizer.process_frame(frame)
                    description = self.face_recognizer.generate_description(faces)
                    self.speak(description)
            
            # Object detection commands
            elif 'what' in command or 'see' in command:
                ret, frame = self.camera.read()
                if ret:
                    objects = self.object_detector.detect_objects(frame)
                    description = self.object_detector.generate_description(objects)
                    self.speak(description)
            
            # Text reading commands
            elif 'read' in command:
                mode = TextMode.DOCUMENT
                if 'sign' in command:
                    mode = TextMode.SIGN
                elif 'label' in command:
                    mode = TextMode.LABEL
                elif 'display' in command:
                    mode = TextMode.DISPLAY
                
                ret, frame = self.camera.read()
                if ret:
                    text = self.text_reader.read_text(frame, mode=mode)
                    self.speak(text)
            
            # Gesture recognition commands
            elif 'gesture' in command or 'movement' in command:
                self.speak("Watching for gestures. Say stop when done.")
                while True:
                    ret, frame = self.camera.read()
                    if ret:
                        gestures = self.gesture_recognizer.detect_gestures(frame)
                        if gestures:
                            description = self.gesture_recognizer.get_gesture_description(gestures)
                            self.speak(description)
                    
                    # Check for stop command
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # Environment description
            elif 'describe' in command or 'environment' in command:
                ret, frame = self.camera.read()
                if ret:
                    objects = self.object_detector.detect_objects(frame)
                    faces = self.face_recognizer.process_frame(frame)
                    gestures = self.gesture_recognizer.detect_gestures(frame)
                    
                    description = self.generate_environment_description(
                        objects, faces, gestures
                    )
                    self.speak(description)
            
            # Help command
            elif 'help' in command:
                self.speak("""
                    I can help you in several ways:
                    Say 'who' or 'recognize' to identify people,
                    Say 'what' or 'see' to describe objects around you,
                    Say 'read' to read text, with options for signs, labels, or displays,
                    Say 'gesture' to detect hand gestures,
                    Say 'describe environment' for a complete description,
                    Or say 'exit' to close the program.
                """)
            
            # Exit command
            elif 'exit' in command:
                self.speak("Goodbye! Stay safe!")
                self.cleanup()
                return True
                
            return False
            
        except Exception as e:
            self.speak(f"I encountered an error: {str(e)}")
            return False

    def run(self):
        """Main system loop"""
        # Start monitoring thread
        monitoring_thread = threading.Thread(
            target=self.continuous_monitoring,
            daemon=True
        )
        monitoring_thread.start()
        
        # Welcome message
        self.speak("""
            Hello! I'm your integrated assistance system.
            I can help you identify people, describe objects,
            read text, and recognize gestures.
            Say 'help' for available commands.
        """)
        
        # Main interaction loop
        while self.is_running:
            command = self.listen()
            if self.process_command(command):
                break
        
        self.cleanup()

    def cleanup(self):
        """Clean up system resources"""
        self.is_running = False
        
        # Cleanup individual components
        self.face_recognizer.cleanup()
        self.gesture_recognizer.cleanup()
        self.object_detector.cleanup()
        self.text_reader.cleanup()
        
        # Release camera
        if self.camera is not None:
            self.camera.release()
        
        cv2.destroyAllWindows()

def main():
    """Main entry point"""
    try:
        system = IntegratedSystem()
        system.run()
    except Exception as e:
        print(f"Critical error: {str(e)}")
        raise
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()