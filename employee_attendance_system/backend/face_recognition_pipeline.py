#!/usr/bin/env python3
"""
Enhanced Face Detection and Recognition Pipeline
- Using DeepFace for face detection and recognition
- Vector database for face embeddings
- Real-time processing optimized
"""

import cv2
import numpy as np
from deepface import DeepFace
import os
import pickle
import logging
from typing import List, Dict, Optional, Tuple
import time
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionPipeline:
    def __init__(self, 
                 face_db_path: str = "face_database",
                 model_name: str = "VGG-Face",
                 detector_backend: str = "opencv",
                 confidence_threshold: float = 0.6):
        """
        Initialize face recognition pipeline
        
        Args:
            face_db_path: Path to store face database
            model_name: DeepFace model (VGG-Face, Facenet, OpenFace, etc.)
            detector_backend: Face detector (opencv, ssd, dlib, etc.)
            confidence_threshold: Minimum confidence for recognition
        """
        self.face_db_path = face_db_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.confidence_threshold = confidence_threshold
        
        # Create face database directory
        os.makedirs(face_db_path, exist_ok=True)
        
        # Load or initialize face database
        self.face_database = self._load_face_database()
        
        # Initialize face detector
        self.face_cascade = None
        self._init_face_detector()
        
        logger.info(f"FaceRecognitionPipeline initialized with {model_name} model")
    
    def _init_face_detector(self):
        """Initialize OpenCV face detector as fallback"""
        try:
            cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_file)
            logger.info("OpenCV face detector initialized")
        except Exception as e:
            logger.warning(f"Could not initialize face detector: {e}")
    
    def _load_face_database(self) -> Dict:
        """Load face database from disk"""
        db_file = os.path.join(self.face_db_path, "face_database.pkl")
        
        if os.path.exists(db_file):
            try:
                with open(db_file, 'rb') as f:
                    database = pickle.load(f)
                logger.info(f"Loaded face database with {len(database)} entries")
                return database
            except Exception as e:
                logger.error(f"Error loading face database: {e}")
        
        logger.info("Initializing new face database")
        return {}
    
    def _save_face_database(self):
        """Save face database to disk"""
        db_file = os.path.join(self.face_db_path, "face_database.pkl")
        
        try:
            with open(db_file, 'wb') as f:
                pickle.dump(self.face_database, f)
            logger.info("Face database saved successfully")
        except Exception as e:
            logger.error(f"Error saving face database: {e}")
    
    def detect_faces_opencv(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using OpenCV (fast fallback)
        
        Args:
            frame: Input image frame
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces.tolist()
        except Exception as e:
            logger.error(f"OpenCV face detection error: {e}")
            return []
    
    def detect_faces_deepface(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces using DeepFace (more accurate but slower)
        
        Args:
            frame: Input image frame
            
        Returns:
            List of face detection results
        """
        try:
            # DeepFace.extract_faces returns face regions
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            # Convert to bounding box format
            results = []
            for i, face in enumerate(faces):
                if face is not None:
                    # Estimate bounding box from face region
                    h, w = frame.shape[:2]
                    face_h, face_w = face.shape[:2]
                    
                    # This is a simplified approach - DeepFace doesn't directly return bbox
                    results.append({
                        'face_id': i,
                        'face_region': face,
                        'confidence': 0.8  # Default confidence
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"DeepFace detection error: {e}")
            return []
    
    def get_face_encoding(self, face_region: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face encoding using DeepFace
        
        Args:
            face_region: Cropped face image
            
        Returns:
            Face encoding vector or None if failed
        """
        try:
            # Ensure face region is valid
            if face_region.shape[0] < 10 or face_region.shape[1] < 10:
                return None
            
            # Get face embedding
            embedding = DeepFace.represent(
                img_path=face_region,
                model_name=self.model_name,
                enforce_detection=False
            )
            
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            
        except Exception as e:
            logger.error(f"Face encoding error: {e}")
        
        return None
    
    def find_face_match(self, face_encoding: np.ndarray) -> Optional[Dict]:
        """
        Find matching face in database using vector similarity
        
        Args:
            face_encoding: Query face encoding
            
        Returns:
            Match result with employee info and confidence
        """
        if len(self.face_database) == 0:
            return None
        
        best_match = None
        best_similarity = 0
        
        try:
            for employee_id, employee_data in self.face_database.items():
                stored_encoding = employee_data.get('face_encoding')
                
                if stored_encoding is not None:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        [face_encoding], 
                        [stored_encoding]
                    )[0][0]
                    
                    if similarity > best_similarity and similarity > self.confidence_threshold:
                        best_similarity = similarity
                        best_match = {
                            'employee_id': employee_id,
                            'employee_info': employee_data['info'],
                            'confidence': float(similarity),
                            'match_type': 'known'
                        }
            
            return best_match
            
        except Exception as e:
            logger.error(f"Face matching error: {e}")
        
        return None
    
    def add_employee_face(self, employee_info: Dict, face_image: np.ndarray) -> bool:
        """
        Add new employee face to database
        
        Args:
            employee_info: Employee information dict
            face_image: Face image
            
        Returns:
            Success status
        """
        try:
            # Get face encoding
            face_encoding = self.get_face_encoding(face_image)
            
            if face_encoding is None:
                logger.error("Could not generate face encoding")
                return False
            
            employee_id = employee_info.get('employee_id')
            if not employee_id:
                logger.error("Employee ID is required")
                return False
            
            # Store in database
            self.face_database[employee_id] = {
                'info': employee_info,
                'face_encoding': face_encoding,
                'created_at': time.time()
            }
            
            # Save to disk
            self._save_face_database()
            
            logger.info(f"Added employee {employee_id} to face database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding employee face: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, use_deepface: bool = False) -> List[Dict]:
        """
        Process video frame for face detection and recognition
        
        Args:
            frame: Input video frame
            use_deepface: Whether to use DeepFace for detection (slower but more accurate)
            
        Returns:
            List of detection results with bounding boxes and recognition info
        """
        results = []
        
        try:
            # Choose detection method
            if use_deepface:
                faces = self.detect_faces_deepface(frame)
                # Handle DeepFace format
                for face_data in faces:
                    face_region = face_data['face_region']
                    face_encoding = self.get_face_encoding(face_region)
                    
                    if face_encoding is not None:
                        match = self.find_face_match(face_encoding)
                        
                        if match:
                            results.append({
                                'bbox': {'x': 0, 'y': 0, 'width': 100, 'height': 100},  # Placeholder
                                'employee': match['employee_info'],
                                'confidence': match['confidence'],
                                'type': 'known'
                            })
                        else:
                            results.append({
                                'bbox': {'x': 0, 'y': 0, 'width': 100, 'height': 100},  # Placeholder
                                'employee': None,
                                'confidence': 0.5,
                                'type': 'stranger'
                            })
            else:
                # Use OpenCV for faster detection
                faces = self.detect_faces_opencv(frame)
                
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_region = frame[y:y+h, x:x+w]
                    
                    if face_region.size > 0:
                        # Get face encoding
                        face_encoding = self.get_face_encoding(face_region)
                        
                        if face_encoding is not None:
                            match = self.find_face_match(face_encoding)
                            
                            if match:
                                results.append({
                                    'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                                    'employee': match['employee_info'],
                                    'confidence': match['confidence'],
                                    'type': 'known'
                                })
                            else:
                                results.append({
                                    'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                                    'employee': None,
                                    'confidence': 0.4,
                                    'type': 'stranger'
                                })
                        else:
                            # Face detected but no encoding - still show as detection
                            results.append({
                                'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                                'employee': None,
                                'confidence': 0.3,
                                'type': 'stranger'
                            })
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
        
        return results
    
    def get_database_stats(self) -> Dict:
        """Get face database statistics"""
        return {
            'total_employees': len(self.face_database),
            'employees': list(self.face_database.keys()),
            'model_name': self.model_name,
            'detector_backend': self.detector_backend,
            'confidence_threshold': self.confidence_threshold
        }

# Demo function to initialize with sample employees
def initialize_demo_database():
    """Initialize face database with demo employees"""
    pipeline = FaceRecognitionPipeline()
    
    # Sample employees (in real use, you'd add actual face images)
    demo_employees = [
        {
            'employee_id': 'EMP001',
            'name': 'Nguyễn Văn A',
            'department': 'IT',
            'position': 'Software Engineer'
        },
        {
            'employee_id': 'EMP002', 
            'name': 'Trần Thị B',
            'department': 'HR',
            'position': 'HR Manager'
        },
        {
            'employee_id': 'EMP003',
            'name': 'Lê Văn C', 
            'department': 'Finance',
            'position': 'Accountant'
        }
    ]
    
    logger.info(f"Demo database initialized with {len(demo_employees)} employees")
    return pipeline

if __name__ == "__main__":
    # Test the pipeline
    pipeline = initialize_demo_database()
    print("Face Recognition Pipeline ready!")
    print(f"Database stats: {pipeline.get_database_stats()}")