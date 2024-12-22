# src/recognition/gesture_recognizer.py
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

class GestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture detection settings
        self.gesture_history = deque(maxlen=30)  # Store last 30 frames
        self.last_gesture_time = time.time()
        self.no_gesture_threshold = 2.0  # Seconds before auto-stop
        self.last_detected_gesture = None
        
        # Initialize camera
        self.camera = None

    def initialize_camera(self, camera=None):
        if camera is not None:
            self.camera = camera
        else:
            self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Could not access camera")

    def calculate_finger_angles(self, hand_landmarks):
        """Calculate angles between finger joints for gesture recognition"""
        angles = {}
        finger_tips = {
            'thumb': self.mp_hands.HandLandmark.THUMB_TIP,
            'index': self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            'middle': self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            'ring': self.mp_hands.HandLandmark.RING_FINGER_TIP,
            'pinky': self.mp_hands.HandLandmark.PINKY_TIP
        }
        
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        for finger, tip_id in finger_tips.items():
            tip = hand_landmarks.landmark[tip_id]
            angle = np.arctan2(tip.y - wrist.y, tip.x - wrist.x)
            angles[finger] = np.degrees(angle)
        
        return angles

    def detect_gestures(self, frame):
        """
        Enhanced gesture detection with auto-stop feature and
        support for common gestures including waves, greetings, etc.
        """
        current_time = time.time()
        
        # Check for auto-stop
        if (current_time - self.last_gesture_time > self.no_gesture_threshold and 
            self.last_detected_gesture is not None):
            self.last_detected_gesture = None
            return ["gesture_stop"]

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        detected_gestures = []
        
        if results.multi_hand_landmarks:
            self.last_gesture_time = current_time
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate finger positions and angles
                angles = self.calculate_finger_angles(hand_landmarks)
                finger_states = self.get_finger_states(hand_landmarks)
                
                # Detect various gestures
                gesture = self.classify_gesture(angles, finger_states)
                if gesture:
                    detected_gestures.append(gesture)
                
                # Draw landmarks for visual feedback
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        return detected_gestures

    def get_finger_states(self, hand_landmarks):
        """Determine if fingers are extended or not"""
        finger_states = {}
        fingers = {
            'thumb': [4, 3, 2],
            'index': [8, 7, 6],
            'middle': [12, 11, 10],
            'ring': [16, 15, 14],
            'pinky': [20, 19, 18]
        }
        
        for finger, points in fingers.items():
            tip = hand_landmarks.landmark[points[0]]
            mid = hand_landmarks.landmark[points[1]]
            bottom = hand_landmarks.landmark[points[2]]
            
            extended = (tip.y < mid.y < bottom.y)
            finger_states[finger] = extended
        
        return finger_states

    def classify_gesture(self, angles, finger_states):
        """
        Classify the detected hand pose into specific gestures.
        Enhanced to recognize multiple natural gestures.
        """
        # Wave detection (side to side movement)
        if self.detect_wave_motion(angles):
            return "wave"
            
        # Thumbs up
        if (finger_states['thumb'] and 
            not any(finger_states[f] for f in ['index', 'middle', 'ring', 'pinky'])):
            return "thumbs_up"
            
        # Thumbs down
        if (not finger_states['thumb'] and 
            not any(finger_states[f] for f in ['index', 'middle', 'ring', 'pinky'])):
            return "thumbs_down"
            
        # Peace sign
        if (finger_states['index'] and finger_states['middle'] and 
            not any(finger_states[f] for f in ['ring', 'pinky'])):
            return "peace"
            
        # Open palm (hello/stop)
        if all(finger_states.values()):
            return "open_palm"
            
        # Pointing
        if (finger_states['index'] and 
            not any(finger_states[f] for f in ['middle', 'ring', 'pinky'])):
            return "pointing"
            
        # Namaste (hands pressed together)
        # This requires detecting both hands
        if len(self.gesture_history) >= 2:
            return self.detect_namaste()
            
        return None

    def detect_wave_motion(self, angles):
        """Detect waving motion by analyzing hand movement patterns"""
        self.gesture_history.append(angles['index'])
        if len(self.gesture_history) >= 10:
            # Calculate movement variance
            variance = np.var(list(self.gesture_history))
            return variance > 500  # Threshold for wave detection
        return False

    def detect_namaste(self):
        """Detect namaste gesture (hands pressed together)"""
        if len(self.gesture_history) < 2:
            return None
            
        recent_angles = list(self.gesture_history)[-2:]
        angle_diff = abs(recent_angles[0] - recent_angles[1])
        
        return "namaste" if angle_diff < 20 else None

    def get_gesture_description(self, gestures):
        """
        Convert detected gestures into natural language descriptions
        suitable for blind users.
        """
        descriptions = {
            "wave": "someone is waving",
            "thumbs_up": "a thumbs up, indicating approval",
            "thumbs_down": "a thumbs down, indicating disapproval",
            "peace": "a peace sign",
            "open_palm": "an open palm, possibly saying hello or stop",
            "pointing": "someone is pointing",
            "namaste": "someone is greeting with namaste",
            "gesture_stop": "no gestures currently detected"
        }
        
        if not gestures:
            return "No gestures detected"
            
        return ". ".join(descriptions.get(g, "Unknown gesture") for g in gestures)

    def cleanup(self):
        """Clean up resources"""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()