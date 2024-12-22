# src/recognition/text_reader.py
import cv2
import pytesseract
import numpy as np
from enum import Enum
import time

class TextMode(Enum):
    DOCUMENT = "document"
    SIGN = "sign"
    LABEL = "label"
    DISPLAY = "display"  # For digital displays, screens
    SCENE = "scene"     # For text in natural scenes

class TextReader:
    def __init__(self):
        """
        Enhanced text reader with specialized modes for different text types
        and improved accuracy for blind users.
        """
        # Configure Tesseract path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Initialize camera
        self.camera = None
        
        # Configure mode-specific settings
        self.mode_configs = {
            TextMode.DOCUMENT: {
                'psm': 3,  # Fully automatic page segmentation
                'preprocessing': self.preprocess_document
            },
            TextMode.SIGN: {
                'psm': 11,  # Sparse text - find as much text as possible
                'preprocessing': self.preprocess_sign
            },
            TextMode.LABEL: {
                'psm': 6,  # Uniform block of text
                'preprocessing': self.preprocess_label
            },
            TextMode.DISPLAY: {
                'psm': 7,  # Treat the image as a single text line
                'preprocessing': self.preprocess_display
            },
            TextMode.SCENE: {
                'psm': 12,  # Sparse text with OSD
                'preprocessing': self.preprocess_scene
            }
        }
        
        # Text filtering and validation settings
        self.min_confidence = 60
        self.last_successful_read = None
        self.read_history = []

    def initialize_camera(self, camera=None):
        if camera is not None:
            self.camera = camera
        else:
            self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Could not access camera")

    def preprocess_document(self, image):
        """
        Optimize image for document text reading.
        Good for printed documents, books, letters.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return threshold

    def preprocess_sign(self, image):
        """
        Optimize image for sign reading.
        Handles varying lighting and angles.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        blur = cv2.GaussianBlur(contrast, (5, 5), 0)
        threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return threshold

    def preprocess_label(self, image):
        """
        Optimize image for label reading.
        Handles curved surfaces and reflections.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
        thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh

    def preprocess_display(self, image):
        """
        Optimize image for digital display reading.
        Handles glare and screen reflections.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright = cv2.convertScaleAbs(gray, alpha=1.3, beta=40)
        blur = cv2.GaussianBlur(bright, (3, 3), 0)
        return blur

    def preprocess_scene(self, image):
        """
        Optimize image for natural scene text.
        Handles varying backgrounds and lighting.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return enhanced

    def read_text(self, frame, mode=TextMode.DOCUMENT, verbose=True):
        """
        Read text from frame with enhanced accuracy and
        mode-specific optimization.
        """
        # Get mode-specific settings
        config = self.mode_configs[mode]
        
        # Preprocess image
        processed = config['preprocessing'](frame)
        
        # Show processing feedback
        if verbose:
            cv2.imshow('Processing Text', processed)
            cv2.waitKey(1)
        
        try:
            # Configure Tesseract
            custom_config = f'--psm {config["psm"]} --oem 3'
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(processed, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Extract text with confidence filtering
            text_results = []
            confidences = []
            
            for i, conf in enumerate(data['conf']):
                if conf > self.min_confidence:
                    text = data['text'][i].strip()
                    if text:
                        text_results.append(text)
                        confidences.append(conf)
            
            # Combine results
            if text_results:
                combined_text = ' '.join(text_results)
                avg_confidence = sum(confidences) / len(confidences)
                
                # Update reading history
                self.read_history.append({
                    'text': combined_text,
                    'confidence': avg_confidence,
                    'timestamp': time.time()
                })
                
                # Clean old history
                self.clean_history()
                
                return self.format_text_result(combined_text, mode)
            
            return "No text detected"
            
        except Exception as e:
            return f"Error reading text: {str(e)}"
        finally:
            if verbose:
                cv2.destroyAllWindows()

    def format_text_result(self, text, mode):
        """
        Format the detected text based on its type and context.
        """
        if mode == TextMode.SIGN:
            return f"Sign reads: {text}"
        elif mode == TextMode.LABEL:
            return f"Label says: {text}"
        elif mode == TextMode.DISPLAY:
            return f"Display shows: {text}"
        elif mode == TextMode.SCENE:
            return f"Detected text: {text}"
        else:  # Document mode
            return text

    def clean_history(self):
        """
        Clean up old reading history.
        Keep only last 5 seconds of readings.
        """
        current_time = time.time()
        self.read_history = [
            h for h in self.read_history
            if current_time - h['timestamp'] <= 5
        ]

    def get_text_mode(self, frame):
        """
        Automatically determine the best text reading mode
        based on image analysis.
        """
        # Analyze image characteristics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Check for document characteristics
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges)
        
        if edge_density < 10:
            return TextMode.DISPLAY
        elif edge_density < 30:
            return TextMode.SIGN
        elif edge_density < 50:
            return TextMode.LABEL
        else:
            return TextMode.DOCUMENT

    def read_continuous(self, duration=5):
        """
        Continuously read text for a specified duration,
        useful for scanning documents or environments.
        """
        start_time = time.time()
        combined_results = set()
        
        while (time.time() - start_time) < duration:
            ret, frame = self.camera.read()
            if not ret:
                continue
                
            # Auto-detect mode and read text
            mode = self.get_text_mode(frame)
            result = self.read_text(frame, mode, verbose=False)
            
            if result != "No text detected":
                combined_results.add(result)
            
            # Show processing feedback
            cv2.imshow('Scanning', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return list(combined_results)

    def cleanup(self):
        """Clean up resources"""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()