
---

# **SpeX** - AI-Powered Smart Glasses for Visually Impaired

**SpeX** is an AI-powered smart glasses solution designed to help visually impaired individuals navigate the world independently. By embedding advanced AI technologies, SpeX acts as the "eyes" for its users, providing real-time visual assistance and auditory feedback for enhanced mobility, safety, and social interaction.

---

## **Problem Statement**

Visually impaired individuals face significant challenges in navigating their environment, interacting with others, and accessing textual information. Traditional tools lack real-time visual assistance, limiting independence. SpeX addresses these challenges by integrating AI-driven object detection, face recognition, text-to-speech conversion, and more, all within a wearable smart glasses system.

---

## **Objective**

SpeX aims to empower visually impaired users by providing a wearable solution that offers:

- Real-time object detection
- Face recognition for social interactions
- Text recognition and reading
- Voice-based interaction for seamless assistance

This AI-driven system promotes independence, safety, and social integration.

---

## **Key Features**

- **Object Detection (YOLO)**: Real-time identification and localization of objects in the environment, helping users navigate obstacles and identify points of interest.
- **Face Recognition (OpenCV)**: Detects and recognizes faces for improved social interaction and safety.
- **Text Recognition (Tesseract OCR)**: Converts printed text (from signs, labels, books, etc.) into spoken words.
- **Voice Interaction**: Seamlessly integrates voice commands to allow users to interact with the assistant hands-free, enabling a natural experience.
- **AI Assistant**: Provides contextual auditory feedback and supports multilingual communication for personalized assistance.

---

## **Technologies Used**

### **Hardware**
- **Webcam**: Captures real-time visual input.
- **Microphone**: Captures voice commands for interaction.
- **Speaker/Headphones**: Outputs auditory feedback.
- **Raspberry Pi 0**: Powers the system with cost-effective hardware.
- **Bluetooth Earphones**: For portable, hands-free audio.

### **Software**
- **Numpy**: Numerical computations and data manipulation.
- **OpenCV**: Real-time image and video processing for face and object recognition.
- **Mediapipe**: Provides real-time ML-based detection and tracking of gestures and faces.
- **Pytesseract**: Extracts text from images for real-time reading.
- **Transformers**: Pre-trained models for NLP tasks such as text summarization.
- **Torch**: Deep learning framework used for training and implementing AI models.
- **Ultralytics**: YOLO-based object detection to help identify objects in real-time.
- **Face_recognition**: Detects and recognizes faces using deep learning models.
- **Speechrecognition**: Converts speech to text, enabling voice command processing.
- **Pytz**: Handles time zone operations for real-time assistance.
- **Geopy**: Provides geolocation and mapping functionalities.
- **Pyttsx3**: Converts text to speech for delivering auditory feedback.

---

## **Installation & Setup**

1. Clone the repository:
   ```bash
   git clone https://github.com/harshendram/speXweb.git
   ```

2. Navigate to the project directory:
   ```bash
   cd speXweb
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python speX.py
   ```

---

## **Usage**

SpeX provides real-time assistance through voice commands. Here's how you can interact with the system:

- **"What is the distance to [location]?"** – Get distance-based queries.
- **"Identify objects around me."** – Detect objects in the user's vicinity.
- **"Who is that?"** – Recognize and identify faces.
- **"Read this text."** – Convert visible text into speech.

---

## **Impact**

SpeX greatly enhances the independence of visually impaired individuals by providing:

- **Autonomous Navigation**: Users can avoid obstacles and identify objects in real-time.
- **Social Interaction**: With face recognition, SpeX helps users recognize friends, family, or others.
- **Information Access**: Text recognition empowers users to read printed materials, signs, and more.
- **Emergency Awareness**: Provides situational awareness, enabling better decision-making in various environments.

---

## **Scalability**

SpeX is built to scale and evolve:

- **Language Expansion**: The AI assistant can be extended to support more languages.
- **Advanced Recognition**: Future improvements in AI models will lead to better accuracy in object and face detection.
- **Assistive Technology Integration**: SpeX can integrate with other assistive technologies to broaden its capabilities.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## **Links**

- [GitHub Repository](https://github.com/harshendram/speXweb)

---

This README is structured to provide a clean, professional overview with modern formatting. It includes everything necessary for someone new to understand the project, install it, and explore its impact. Let me know if you'd like to add or modify anything!
