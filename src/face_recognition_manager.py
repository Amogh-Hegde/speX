import cv2
import face_recognition
import pickle
import numpy as np
from datetime import datetime

class FaceRecognitionManager:
    def __init__(self, model_path='models/face_recognition_data.pkl'):
        """
        This class manages real-time face recognition during system operation.
        It loads  trained face data and provides natural language descriptions
        of recognized people, which is especially important for blind users.
        """

        # Initialize the recognition system
        self.model_path = model_path
        
        # Load  trained face data
        print("Loading face recognition data...")
        self.load_face_data()
        
        # Configure recognition parameters
        self.recognition_threshold = 0.6  # Lower means stricter matching
        self.last_recognition_time = {}  # To prevent repetitive announcements
        self.announcement_cooldown = 5  # Seconds between announcements for same person

        self.camera = None  # Initialize camera to None, so we can set it later

    def load_face_data(self):
        """
        Loads the trained face recognition data from  saved model file.
        This includes the face encodings and associated names/relationships.
        """
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
                self.known_face_relations = data['relations']
                print("Face recognition data loaded successfully!")
        except FileNotFoundError:
            print("No face recognition data found. Please run training first.")
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_relations = []

    def initialize_camera(self, camera_device=0):
        """
        Initialize the camera to be used for real-time face recognition.
        """
        if camera_device is None:
            camera_device = 0  # Default to the first camera

        print(f"Initializing camera with device index: {camera_device}")
        try:
            self.camera = cv2.VideoCapture(camera_device, cv2.CAP_DSHOW)  # Use DirectShow for Windows
            if not self.camera.isOpened():
                raise ValueError("Camera device could not be opened. Please check the index or permissions.")
            print("Camera initialized successfully.")
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.camera = None

    def process_frame(self, frame):
        """
        Processes a video frame to recognize faces in real-time.
        Returns natural language descriptions of who was recognized.
        """
        # Convert frame from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = frame[:, :, ::-1]
        
        # Find all faces in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        current_time = datetime.now()
        recognitions = []
        
        if len(face_encodings) == 0:
            print("No faces detected.")
        
        # Process each detected face
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(
                self.known_face_encodings,
                face_encoding,
                tolerance=self.recognition_threshold
            )
            
            if True in matches:
                # Find the best match
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings,
                    face_encoding
                )
                best_match_index = np.argmin(face_distances)
                
                name = self.known_face_names[best_match_index]
                relation = self.known_face_relations[best_match_index]
                
                # Check if we should announce this person again
                last_time = self.last_recognition_time.get(name)
                if (not last_time or 
                    (current_time - last_time).total_seconds() > self.announcement_cooldown):
                    
                    # Create a natural description
                    if relation in ['mom', 'dad', 'brother', 'sister']:
                        description = f"your {relation} {name}"
                    else:
                        description = f"{name}, who is {relation}"
                    
                    recognitions.append({
                        'name': name,
                        'relation': relation,
                        'description': description,
                        'location': face_location
                    })
                    
                    # Update last recognition time
                    self.last_recognition_time[name] = current_time
            else:
                recognitions.append({
                    'name': 'Unknown',
                    'relation': None,
                    'description': "someone I don't recognize",
                    'location': face_location
                })
        
        return recognitions

    def generate_description(self, recognitions):
        """
        Creates a natural language description of who was recognized.
        Focuses on making the descriptions helpful for blind users.
        """
        if not recognitions:
            return "I don't see any faces right now."
        
        descriptions = []
        known_people = [r for r in recognitions if r['name'] != 'Unknown']
        unknown_count = len([r for r in recognitions if r['name'] == 'Unknown'])
        
        if known_people:
            people_desc = [r['description'] for r in known_people]
            if len(people_desc) == 1:
                descriptions.append(f"I see {people_desc[0]}")
            else:
                descriptions.append(f"I see {', '.join(people_desc[:-1])} and {people_desc[-1]}")

        if unknown_count:
            if unknown_count == 1:
                descriptions.append("and someone I don't recognize")
            else:
                descriptions.append(f"and {unknown_count} people I don't recognize")
        
        return " ".join(descriptions)

    def draw_recognitions(self, frame, recognitions):
        """
        Draws bounding boxes and labels on the frame for recognized faces.
        This is helpful for sighted users assisting with the system.
        """
        for recognition in recognitions:
            top, right, bottom, left = recognition['location']
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw a label with the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, recognition['name'], (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
