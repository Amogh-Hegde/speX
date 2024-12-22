# src/recognition/face_trainer.py
import os
import cv2
import numpy as np
import pickle
import face_recognition
from datetime import datetime
import logging

class FaceTrainingSystem:
    def __init__(self, dataset_path='datasets/faces'):
        """
        This system handles the training of our face recognition model.
        Think of it like teaching the system to recognize faces the same way
        a person learns to recognize their family members - through repeated
        exposure and learning distinctive features.
        """
        # Set up file paths
        self.dataset_path = dataset_path
        self.model_path = 'models/face_recognition_data.pkl'
        
        # Initialize storage for face data
        self.known_face_encodings = []  # Will store the unique features of each face
        self.known_face_names = []      # Will store the names corresponding to each face
        self.known_face_relations = []   # Will store relationships (mom, dad, etc.)
        
        # Set up logging to track training process
        self.setup_logging()
        
        self.logger.info("Face Training System initialized")

    def setup_logging(self):
        """
        Sets up detailed logging so we can track what happens during training.
        This is especially important for debugging if something goes wrong.
        """
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'face_training_{datetime.now():%Y%m%d_%H%M%S}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('FaceTraining')

    def process_person_images(self, person_dir):
        """
        Processes all images for a single person, creating face encodings
        that will be used for recognition. This is like learning all the
        different ways one person can look from different angles.
        """
        name, relation = person_dir.split('_')
        person_path = os.path.join(self.dataset_path, person_dir)
        successful_encodings = 0
        
        self.logger.info(f"Processing images for {name} ({relation})")
        
        for image_file in os.listdir(person_path):
            if not image_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            image_path = os.path.join(person_path, image_file)
            try:
                # Load the image and find all faces in it
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    # Store the face encoding and associated information
                    self.known_face_encodings.append(encodings[0])
                    self.known_face_names.append(name)
                    self.known_face_relations.append(relation)
                    successful_encodings += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {image_file}: {str(e)}")
                continue
        
        return successful_encodings

    def train(self):
        """
        Main training function that processes entire dataset of face images.
        This combines all the individual face encodings into a single model
        that can recognize everyone trained upon.
        """
        self.logger.info("Starting face recognition training")
        
        total_processed = 0
        training_summary = []
        
        # Process each person's directory
        for person_dir in os.listdir(self.dataset_path):
            if not os.path.isdir(os.path.join(self.dataset_path, person_dir)):
                continue
            
            processed_count = self.process_person_images(person_dir)
            total_processed += processed_count
            
            # Record the training results for this person
            name, relation = person_dir.split('_')
            training_summary.append({
                'name': name,
                'relation': relation,
                'images_processed': processed_count,
                'timestamp': datetime.now().isoformat()
            })
        
        # Save trained model
        self.save_model(training_summary)
        
        self.logger.info(f"Training completed. Processed {total_processed} images total")
        return total_processed

    def save_model(self, training_summary):
        """
        Saves our trained face recognition model and its training history.
        This lets us reload the trained model without having to retrain.
        """
        model_data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names,
            'relations': self.known_face_relations,
            'training_summary': training_summary,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {self.model_path}")

def main():
    """
    Main function that runs our training process and provides
    feedback about the results.
    """
    trainer = FaceTrainingSystem()
    
    print("\nFace Recognition Training System")
    print("--------------------------------")
    print("This system will process all face images in your dataset")
    print("and train the recognition model to identify your family and friends.")
    
    input("Press Enter to start training...")
    
    try:
        total_processed = trainer.train()
        print("\nTraining completed successfully!")
        print(f"Processed {total_processed} images")
        print("The system is now ready to recognize these people.")
        
    except Exception as e:
        print(f"\nAn error occurred during training: {str(e)}")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    main()