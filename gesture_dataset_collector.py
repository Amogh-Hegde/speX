import cv2
import os
import mediapipe as mp

def collect_gesture_data():
    """
    Collects gesture data using webcam.
    Saves images of different gestures for training.
    """
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Set environment variable

    gestures = ['thumbs_up', 'thumbs_down', 'wave', 'peace', 'namaste']
    base_dir = 'training_data/raw_data/gestures'
    os.makedirs(base_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access webcam")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    drawing_utils = mp.solutions.drawing_utils

    for gesture in gestures:
        os.makedirs(os.path.join(base_dir, gesture), exist_ok=True)
        print(f"Collecting data for {gesture} gesture")
        print("Press 'c' to capture, 'n' for next gesture, 'q' to quit")

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame")
                break

            # Process the frame with MediaPipe
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Collect Gestures', frame)
            key = cv2.waitKey(1)

            if key == ord('c'):
                # Save frame
                cv2.imwrite(os.path.join(base_dir, gesture, f"{count}.jpg"), frame)
                count += 1
                print(f"Captured {count} images for {gesture}")
            elif key == ord('n'):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_gesture_data()
