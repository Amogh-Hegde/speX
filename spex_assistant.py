import time
import cv2
import pyttsx3
from recognition.object_detector import ObjectDetector
from text_reader import TextReader
from face_recognition_manager import FaceRecognitionManager
from recognition.gesture_recognizer import GestureRecognizer
from scene_analyzer import SceneAnalyzer
from threading import Thread

class speXAssistant:
    def __init__(self):
        """
        Initialize the speX system, combining all features like object detection,
        text reading, face recognition, gesture recognition, and scene analysis.
        """
        # Initialize individual components
        self.object_detector = ObjectDetector()
        self.text_reader = TextReader()
        self.face_recognition = FaceRecognitionManager()
        self.gesture_recognizer = GestureRecognizer()
        self.scene_analyzer = SceneAnalyzer()

        # Initialize camera for all components
        self.object_detector.initialize_camera()
        self.text_reader.initialize_camera()

        # Initialize the text-to-speech engine for accessibility
        self.speech_engine = pyttsx3.init()
        self.speech_engine.setProperty('rate', 150)  # Adjust speech rate
        self.speech_engine.setProperty('volume', 1)  # Set volume level (0.0 to 1.0)

        # Initialize a variable to hold results
        self.latest_description = ""

    def get_scene_description(self):
        """
        Capture frames and get a detailed scene description by combining results from object detection,
        text reading, face recognition, gesture recognition, and scene analysis.
        """
        print("Capturing and analyzing scene...")

        # Capture frame from the camera
        ret, frame = self.object_detector.camera.read()
        if not ret:
            return "Error accessing camera."

        # Get object detection description
        object_description = self.object_detector.get_scene_description()

        # Get text description using TextReader
        text_description = self.text_reader.read_text()

        # Get face recognition description
        face_description = self.face_recognition.recognize_faces(frame)

        # Get gesture recognition description
        gesture_description = self.gesture_recognizer.recognize_gesture(frame)

        # Analyze scene for additional context
        scene_description = self.scene_analyzer.analyze_scene(frame)

        # Combine all descriptions into one string
        full_description = (f"Scene Description: {object_description}. "
                            f"Detected text: {text_description}. "
                            f"Recognized faces: {face_description}. "
                            f"Gesture recognition: {gesture_description}. "
                            f"Additional scene context: {scene_description}")

        return full_description

    def speak(self, text):
        """
        Use text-to-speech to read out the description for the user.
        """
        self.speech_engine.say(text)
        self.speech_engine.runAndWait()

    def run(self):
        """
        Run the speX system, continuously processing and providing scene descriptions.
        """
        try:
            while True:
                # Get the full scene description (objects, text, faces, gestures, etc.)
                description = self.get_scene_description()
                print(description)

                # Provide spoken feedback for accessibility
                self.speak(description)

                # Store the latest description for future use (e.g., for analysis)
                self.latest_description = description

                # Wait before the next frame
                time.sleep(2)  # Adjust the delay as needed

        except KeyboardInterrupt:
            print("Exiting speX system...")
        finally:
            # Clean up resources after use
            self.cleanup()

    def cleanup(self):
        """
        Clean up all resources after use.
        """
        self.object_detector.cleanup()
        self.text_reader.cleanup()
        self.speech_engine.stop()

if __name__ == "__main__":
    # Create the speX assistant and run it
    spex = speXAssistant()
    spex.run()
