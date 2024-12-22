import face_recognition
import cv2
import numpy as np
import os
import pickle
from datetime import datetime

class FaceRecognizer:
    def __init__(self, known_faces_dir="known_faces"):
        """
        Initialize facial recognition system using the face_recognition library,
        which provides high-accuracy face detection and recognition.
        """
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """
        Loads and encodes faces from the known_faces directory.
        Each person should have their own subdirectory with their name.
        """
        # Check if we have cached encodings
        if os.path.exists('face_encodings.pkl'):
            with open('face_encodings.pkl', 'rb') as f:
                cache = pickle.load(f)
                self.known_face_encodings = cache['encodings']
                self.known_face_names = cache['names']
                return

        # Load face images and encode them
        for person_dir in os.listdir(self.known_faces_dir):
            person_path = os.path.join(self.known_faces_dir, person_dir)
            if os.path.isdir(person_path):
                for image_file in os.listdir(person_path):
                    if image_file.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(person_path, image_file)
                        self._encode_face(image_path, person_dir)

        # Cache the encodings
        with open('face_encodings.pkl', 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)

    def _encode_face(self, image_path, person_name):
        """
        Encodes a single face image and adds it to known faces.
        """
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(person_name)
                
        except Exception as e:
            print(f"Error encoding face in {image_path}: {str(e)}")

    def recognize_faces(self, frame):
        """
        Recognizes faces in a given frame and returns their names
        and locations. Optimized for real-time processing.
        """
        # Resize frame for faster face recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert BGR to RGB
        rgb_small_frame = small_frame[:, :, ::-1]
        
        # Find faces in frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        face_details = []
        
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Match face with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding,
                tolerance=0.6
            )
            name = "Unknown"
            confidence = 0.0
            
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
                confidence = 1 - min(face_recognition.face_distance(
                    [self.known_face_encodings[first_match_index]], 
                    face_encoding
                ))

            # Log recognition event
            if name != "Unknown":
                print(f"[{datetime.now()}] Recognized: {name} (Confidence: {round(confidence, 2)})")

            # Scale back up face locations
            face_details.append({
                'name': name,
                'confidence': round(confidence, 2),
                'location': (
                    top * 4, right * 4,
                    bottom * 4, left * 4
                )
            })
            face_names.append(name)
        
        return face_details

    def add_new_face(self, image, name):
        """
        Adds a new face to the known faces database.
        """
        # Create person directory if it doesn't exist
        person_dir = os.path.join(self.known_faces_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Save image
        image_path = os.path.join(person_dir, f"{name}_{len(os.listdir(person_dir))}.jpg")
        cv2.imwrite(image_path, image)
        
        # Encode and add to known faces
        self._encode_face(image_path, name)
        
        # Update cache
        with open('face_encodings.pkl', 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)

    def remove_face(self, name):
        """
        Removes a person's face data from the system.
        """
        if name in self.known_face_names:
            index = self.known_face_names.index(name)
            del self.known_face_encodings[index]
            del self.known_face_names[index]
            
            # Remove their directory
            person_dir = os.path.join(self.known_faces_dir, name)
            if os.path.exists(person_dir):
                for file in os.listdir(person_dir):
                    os.remove(os.path.join(person_dir, file))
                os.rmdir(person_dir)
            
            # Update cache
            with open('face_encodings.pkl', 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)

    def list_known_faces(self):
        """
        Returns a list of all known faces.
        """
        return self.known_face_names

    def clear_cache(self):
        """
        Clears all cached data and known faces.
        """
        self.known_face_encodings = []
        self.known_face_names = []
        if os.path.exists('face_encodings.pkl'):
            os.remove('face_encodings.pkl')
        if os.path.exists(self.known_faces_dir):
            for person_dir in os.listdir(self.known_faces_dir):
                person_path = os.path.join(self.known_faces_dir, person_dir)
                if os.path.isdir(person_path):
                    for file in os.listdir(person_path):
                        os.remove(os.path.join(person_path, file))
                    os.rmdir(person_path)
