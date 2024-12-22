import cv2
import speech_recognition as sr
import pyttsx3
import datetime
import time
from text_reader import TextReader
from recognition.object_detector import ObjectDetector
from recognition.gesture_recognizer import GestureRecognizer

class ManjuAssistant:
    def __init__(self):
        """
        Initialize Manju with all its capabilities.
        The assistant is designed specifically to help blind individuals
        interact with their environment.
        """
        print("Starting Manju initialization...")
        
        # Initialize voice components first - our primary way to communicate
        print("Setting up speech systems...")
        self.recognizer = sr.Recognizer()
        self.speaker = pyttsx3.init()
        self.speaker.setProperty('rate', 150)
        self.speaker.setProperty('volume', 0.9)
        print("Speech systems ready")

        # Initialize the specialized modules one by one
        try:
            # Text reading capabilities
            print("Initializing text reading system...")
            self.text_reader = TextReader()
            
            # Object detection capabilities
            print("Initializing object detection system...")
            self.object_detector = ObjectDetector()
            
            # Gesture recognition capabilities
            print("Initializing gesture recognition system...")
            self.gesture_recognizer = GestureRecognizer()
            
            # Initialize shared camera resource
            self.initialize_camera()
            
            print("All systems initialized successfully!")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def initialize_camera(self):
        """
        Set up and share a single camera instance among all modules
        to prevent resource conflicts.
        """
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Could not access camera")
            
        # Share camera with all modules
        self.text_reader.initialize_camera(self.camera)
        self.object_detector.initialize_camera(self.camera)
        self.gesture_recognizer.initialize_camera(self.camera)

    def speak(self, text):
        """Convert text to speech with clear pronunciation"""
        print(f"Manju: {text}")
        self.speaker.say(text)
        self.speaker.runAndWait()

    def listen(self):
        """Listen for voice commands with improved error handling"""
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                print("Processing speech...")
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text.lower()
            except sr.WaitTimeoutError:
                print("No speech detected")
                return ""
            except sr.UnknownValueError:
                print("Could not understand audio")
                return ""
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return ""

    def process_command(self, command):
        """
        Process voice commands and coordinate between different modules.
        Each command is handled by the appropriate module while maintaining
        a smooth user experience.
        """
        if not command:
            return False
            
        try:
            # Basic interaction commands
            if 'hello' in command:
                self.speak("Hello! I am Manju. I can help you read text, identify objects, and recognize gestures. How can I assist you?")
            
            elif 'time' in command:
                time = datetime.datetime.now().strftime("%H:%M")
                self.speak(f"The current time is {time}")
            
            # Text reading commands
            elif any(word in command for word in ['read', 'text', 'document', 'label']):
                self.speak("I'll help you read. Please hold the text steady.")
                
                if 'document' in command:
                    text = self.text_reader.read_text('document')
                elif 'label' in command:
                    text = self.text_reader.read_text('label')
                else:
                    text = self.text_reader.read_text()
                    
                self.speak(f"Here's what I read: {text}")
            
            # Object detection commands
            elif any(word in command for word in ['see', 'look', 'what', 'objects']):
                self.speak("Let me look around and describe what I see.")
                scene_description = self.object_detector.get_scene_description()
                self.speak(scene_description)
            
            # Gesture recognition commands
            elif any(word in command for word in ['gesture', 'movement', 'hand']):
                self.speak("I'll watch for gestures. I'll describe any gestures I recognize.")
                
                # Monitor gestures for a set period or until stopped
                start_time = time.time()
                while time.time() - start_time < 30:  # Monitor for 30 seconds
                    gestures = self.gesture_recognizer.recognize_gestures()
                    if gestures:
                        description = self.gesture_recognizer.get_gesture_description(gestures)
                        self.speak(description)
                    
                    # Check for stop command
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                self.speak("Finished gesture recognition mode.")
            
            # Environment awareness commands
            elif 'surroundings' in command or 'around me' in command:
                self.speak("I'll give you a complete description of your surroundings.")
                
                # Get information from all modules
                objects = self.object_detector.get_scene_description()
                gestures = self.gesture_recognizer.recognize_gestures()
                gesture_description = self.gesture_recognizer.get_gesture_description(gestures)
                
                # Provide comprehensive description
                self.speak(f"In your surroundings: {objects}")
                if gesture_description != "No gestures detected":
                    self.speak(f"I also notice: {gesture_description}")
            
            # Help command
            elif 'help' in command:
                self.speak("""
                    I can help you in several ways:
                    Say 'read' to have me read text,
                    Say 'look' or 'see' to have me describe objects around you,
                    Say 'gesture' to have me recognize hand gestures,
                    Say 'surroundings' for a complete description of your environment,
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

    def cleanup(self):
        """Properly clean up all resources"""
        try:
            self.text_reader.cleanup()
            self.object_detector.cleanup()
            self.gesture_recognizer.cleanup()
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

def main():
    """
    Main function to run the Manju assistant.
    Provides a continuous interaction loop with proper error handling.
    """
    try:
        assistant = ManjuAssistant()
        assistant.speak("Hello! I am Manju, your assistant. I can help you read text, identify objects, and recognize gestures. Say 'help' for available commands.")
        
        while True:
            command = assistant.listen()
            if assistant.process_command(command):
                break
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()