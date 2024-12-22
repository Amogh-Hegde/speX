# face_dataset_builder.py
import cv2
import os
from datetime import datetime

class PersonalFaceCollector:
    def __init__(self):
        """
        Initialize  face collection system. This will help build a dataset
        of faces for people you want the system to recognize.
        """
        # Create  directory structure
        self.dataset_dir = "datasets/faces"
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)

    def add_new_person(self, person_name, relationship):
        """
        Add a new person to the dataset. For example:
        add_new_person("John", "brother")
        """
        # Create directory for this person
        person_dir = os.path.join(self.dataset_dir, f"{person_name}_{relationship}")
        os.makedirs(person_dir, exist_ok=True)
        
        print(f"Adding photos for {person_name} ({relationship})")
        print("Position the person's face in front of the camera")
        print("Press 'c' to capture a photo (try different angles)")
        print("Press 'q' when done")
        
        photo_count = 0
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
                
            # Show live camera feed
            cv2.imshow('Add New Person', frame)
            
            key = cv2.waitKey(1)
            if key == ord('c'):
                # Detect face and save it
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    # Save the face image
                    face = frame[y:y+h, x:x+w]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{person_name}_{timestamp}_{photo_count}.jpg"
                    cv2.imwrite(os.path.join(person_dir, filename), face)
                    photo_count += 1
                    print(f"Captured photo {photo_count}")
            
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print(f"Added {photo_count} photos for {person_name}")

    def update_existing_person(self, person_name):
        """
        Add more photos to an existing person's dataset.
        Useful for improving recognition over time.
        """
        # Find the person's directory
        person_dirs = [d for d in os.listdir(self.dataset_dir) 
                      if d.startswith(person_name + "_")]
        
        if not person_dirs:
            print(f"No dataset found for {person_name}")
            return
        
        person_dir = os.path.join(self.dataset_dir, person_dirs[0])
        print(f"Adding more photos for {person_name}")
        # Rest of the code similar to add_new_person...

    def cleanup(self):
        """Release camera resources"""
        self.camera.release()
        cv2.destroyAllWindows()

def main():
    collector = PersonalFaceCollector()
    
    while True:
        print("\nPersonal Face Dataset Builder")
        print("1. Add new person")
        print("2. Update existing person")
        print("3. Exit")
        
        choice = input("Choose an option: ")
        
        if choice == "1":
            name = input("Enter person's name: ")
            relation = input("Enter relationship (mom/dad/brother/etc): ")
            collector.add_new_person(name, relation)
        elif choice == "2":
            name = input("Enter person's name to update: ")
            collector.update_existing_person(name)
        elif choice == "3":
            break
    
    collector.cleanup()

if __name__ == "__main__":
    main()