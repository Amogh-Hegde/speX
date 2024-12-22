# src/recognition/object_detector.py
import cv2
import numpy as np
import time

class ObjectDetector:
    def __init__(self, models_path="models"):
        """
        Enhanced object detector with better spatial awareness and
        natural descriptions for blind users.
        """
        print("Initializing Object Detection System...")
        
        # Load YOLO network
        self.net = cv2.dnn.readNet(
            f"{models_path}/yolov3.weights",
            f"{models_path}/yolov3.cfg"
        )
        
        # Load object classes
        with open(f"{models_path}/coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layers
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Initialize camera
        self.camera = None
        
        # Object tracking settings
        self.last_detections = {}
        self.tracking_history = {}
        
        # Define priority objects for blind users
        self.priority_objects = {
            'high': ['person', 'car', 'truck', 'traffic light', 'stop sign', 'door'],
            'medium': ['chair', 'table', 'stairs', 'bed', 'couch'],
            'low': ['cup', 'bottle', 'book', 'cell phone']
        }

    def initialize_camera(self, camera=None):
        if camera is not None:
            self.camera = camera
        else:
            self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Could not access camera")

    def get_location_description(self, x, y, w, h, frame_width, frame_height):
        """
        Convert object location to natural language description.
        """
        center_x = x + w/2
        center_y = y + h/2
        
        # Horizontal position
        if center_x < frame_width/3:
            h_pos = "on the left"
        elif center_x < 2*frame_width/3:
            h_pos = "in the center"
        else:
            h_pos = "on the right"
            
        # Vertical position
        if center_y < frame_height/3:
            v_pos = "top"
        elif center_y < 2*frame_height/3:
            v_pos = "middle"
        else:
            v_pos = "bottom"
            
        # Distance estimation based on object size
        area_ratio = (w * h) / (frame_width * frame_height)
        if area_ratio > 0.3:
            distance = "very close"
        elif area_ratio > 0.1:
            distance = "nearby"
        else:
            distance = "further away"
            
        return f"{h_pos} {v_pos}, {distance}"

    def detect_objects(self, frame):
        """
        Detect and analyze objects in the frame with enhanced descriptions
        for blind users.
        """
        height, width, _ = frame.shape
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        # Initialize detection lists
        class_ids = []
        confidences = []
        boxes = []
        detections = []
        
        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:  # Confidence threshold
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Process valid detections
        current_detections = {}
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                confidence = confidences[i]
                location = self.get_location_description(x, y, w, h, width, height)
                
                detection = {
                    'label': label,
                    'location': location,
                    'confidence': confidence,
                    'box': boxes[i],
                    'priority': self.get_object_priority(label)
                }
                
                current_detections[label] = detection
                detections.append(detection)
        
        # Update tracking history
        self.update_tracking(current_detections)
        
        return self.generate_description(detections)

    def get_object_priority(self, label):
        """
        Determine priority level of detected object for blind users.
        """
        if label in self.priority_objects['high']:
            return 'high'
        elif label in self.priority_objects['medium']:
            return 'medium'
        elif label in self.priority_objects['low']:
            return 'low'
        return 'normal'

    def update_tracking(self, current_detections):
        """
        Update object tracking history to detect changes in environment.
        """
        current_time = time.time()
        
        # Add new detections
        for label, detection in current_detections.items():
            if label not in self.tracking_history:
                self.tracking_history[label] = []
            self.tracking_history[label].append((current_time, detection))
        
        # Clean up old tracking data
        cleanup_time = current_time - 5  # Keep 5 seconds of history
        for label in list(self.tracking_history.keys()):
            self.tracking_history[label] = [
                (t, d) for t, d in self.tracking_history[label]
                if t > cleanup_time
            ]
            if not self.tracking_history[label]:
                del self.tracking_history[label]

    def generate_description(self, detections):
        """
        Generate natural language description of detected objects,
        prioritizing important information for blind users.
        """
        if not detections:
            return "No objects detected"
        
        # Sort detections by priority
        detections.sort(key=lambda x: x['priority'])
        
        descriptions = []
        
        # Process high priority objects first
        high_priority = [d for d in detections if d['priority'] == 'high']
        if high_priority:
            high_desc = []
            for d in high_priority:
                high_desc.append(f"{d['label']} {d['location']}")
            descriptions.append("Important: " + ", ".join(high_desc))
        
        # Process other objects
        other_objects = [d for d in detections if d['priority'] != 'high']
        if other_objects:
            obj_desc = []
            for d in other_objects:
                obj_desc.append(f"{d['label']} {d['location']}")
            descriptions.append("Also seen: " + ", ".join(obj_desc))
        
        return ". ".join(descriptions)

    def cleanup(self):
        """Clean up resources"""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()