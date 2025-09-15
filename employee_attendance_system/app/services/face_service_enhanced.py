#!/usr/bin/env python3
"""
face_service.py - ENHANCED FACE RECOGNITION SERVICE
Real-time face detection and recognition using DeepFace
Optimized for video streaming with vector comparison
"""

import cv2
import numpy as np
import base64
import os
import logging
import time
import json
from typing import List, Dict, Optional, Tuple
from PIL import Image
import uuid
from datetime import datetime

# Deep learning imports
try:
    from deepface import DeepFace
    from deepface.commons import functions
    DEEPFACE_AVAILABLE = True
except ImportError:
    print("⚠️ DeepFace not available, using basic OpenCV detection")
    DEEPFACE_AVAILABLE = False

# Database imports
from models.models import VectorFace, Employee

logger = logging.getLogger(__name__)

class FaceService:
    """Enhanced face recognition service with real-time processing"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.model_name = 'Facenet512'  # Best accuracy
        self.detector_backend = 'opencv'
        self.threshold = 0.6  # Recognition threshold
        self.face_db = {}  # In-memory face database for speed
        self.detection_cache = {}  # Cache recent detections
        self.stats = {
            'total_recognitions': 0,
            'successful_matches': 0,
            'processing_times': [],
            'cache_hits': 0
        }
        
        # Initialize face detection
        self._load_face_cascade()
        self._load_face_database()
        
        logger.info(f"✅ FaceService initialized with {self.model_name}")
    
    def _load_face_cascade(self):
        """Load OpenCV face cascade for detection"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise Exception("Could not load face cascade")
                
            logger.info("✅ OpenCV face cascade loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load face cascade: {e}")
            self.face_cascade = None
    
    def _load_face_database(self):
        """Load all employee face vectors into memory for fast comparison"""
        try:
            if not self.db_manager:
                logger.warning("⚠️ No database manager, using empty face database")
                return
            
            session = self.db_manager.get_session()
            
            # Load all face vectors with employee data
            vectors = session.query(VectorFace).join(Employee).all()
            
            for vector in vectors:
                employee_id = vector.employee_id
                if employee_id not in self.face_db:
                    self.face_db[employee_id] = {
                        'employee': {
                            'id': vector.employee.id,
                            'name': vector.employee.name,
                            'employee_code': vector.employee.employee_code,
                            'department': vector.employee.department,
                            'position': vector.employee.position
                        },
                        'vectors': []
                    }
                
                # Parse vector data
                try:
                    vector_array = json.loads(vector.vector_data)
                    self.face_db[employee_id]['vectors'].append({
                        'id': vector.id,
                        'vector': np.array(vector_array),
                        'image_path': vector.image_path
                    })
                except Exception as e:
                    logger.error(f"❌ Error parsing vector {vector.id}: {e}")
            
            self.db_manager.close_session(session)
            
            total_employees = len(self.face_db)
            total_vectors = sum(len(emp['vectors']) for emp in self.face_db.values())
            logger.info(f"✅ Loaded {total_vectors} face vectors for {total_employees} employees")
            
        except Exception as e:
            logger.error(f"❌ Error loading face database: {e}")
            self.face_db = {}
    
    def recognize_face(self, frame: np.ndarray) -> List[Dict]:
        """
        Real-time face recognition on video frame
        Returns list of detection results with bounding boxes and employee info
        """
        start_time = time.time()
        results = []
        
        try:
            if frame is None or frame.size == 0:
                return results
            
            # Detect faces using OpenCV (faster than DeepFace for detection)
            faces = self._detect_faces_opencv(frame)
            
            if not faces:
                return results
            
            # Process each detected face
            for (x, y, w, h) in faces:
                try:
                    # Extract face region
                    face_region = frame[y:y+h, x:x+w]
                    
                    if face_region.size == 0:
                        continue
                    
                    # Recognize face if DeepFace is available
                    if DEEPFACE_AVAILABLE and self.face_db:
                        recognition_result = self._recognize_face_deepface(face_region)
                    else:
                        recognition_result = self._create_unknown_result()
                    
                    # Add bounding box info
                    recognition_result['bbox'] = [int(x), int(y), int(w), int(h)]
                    recognition_result['detection_confidence'] = 0.95  # OpenCV detection confidence
                    
                    results.append(recognition_result)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing face region: {e}")
                    continue
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['total_recognitions'] += 1
            self.stats['processing_times'].append(processing_time)
            
            # Keep only last 100 processing times
            if len(self.stats['processing_times']) > 100:
                self.stats['processing_times'].pop(0)
            
            logger.debug(f"🔍 Processed {len(results)} faces in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Face recognition error: {e}")
        
        return results
    
    def _detect_faces_opencv(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV (fast for real-time)"""
        try:
            if self.face_cascade is None:
                return []
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces.tolist() if len(faces) > 0 else []
            
        except Exception as e:
            logger.error(f"❌ OpenCV face detection error: {e}")
            return []
    
    def _recognize_face_deepface(self, face_region: np.ndarray) -> Dict:
        """Recognize face using DeepFace and vector comparison"""
        try:
            # Extract face embedding
            face_embedding = self._extract_face_embedding(face_region)
            
            if face_embedding is None:
                return self._create_unknown_result()
            
            # Find best match in face database
            best_match = self._find_best_match(face_embedding)
            
            if best_match:
                self.stats['successful_matches'] += 1
                return {
                    'type': 'known',
                    'employee': best_match['employee'],
                    'confidence': best_match['confidence'],
                    'vector_id': best_match['vector_id'],
                    'match_method': 'deepface_vector'
                }
            else:
                return self._create_unknown_result()
                
        except Exception as e:
            logger.error(f"❌ DeepFace recognition error: {e}")
            return self._create_unknown_result()
    
    def _extract_face_embedding(self, face_region: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using DeepFace"""
        try:
            # Resize face region for better recognition
            face_resized = cv2.resize(face_region, (160, 160))
            
            # Convert BGR to RGB for DeepFace
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Extract embedding
            embedding = DeepFace.represent(
                img_path=face_rgb,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Face embedding extraction failed: {e}")
            return None
    
    def _find_best_match(self, face_embedding: np.ndarray) -> Optional[Dict]:
        """Find best matching employee using cosine similarity"""
        try:
            best_match = None
            best_distance = float('inf')
            
            for employee_id, employee_data in self.face_db.items():
                for vector_data in employee_data['vectors']:
                    stored_vector = vector_data['vector']
                    
                    # Calculate cosine distance
                    distance = self._cosine_distance(face_embedding, stored_vector)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = {
                            'employee': employee_data['employee'],
                            'confidence': 1 - distance,  # Convert distance to confidence
                            'distance': distance,
                            'vector_id': vector_data['id']
                        }
            
            # Check if best match meets threshold
            if best_match and best_match['distance'] < self.threshold:
                return best_match
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Best match finding error: {e}")
            return None
    
    def _cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine distance between two vectors"""
        try:
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calculate cosine similarity
            cosine_sim = np.dot(vec1_norm, vec2_norm)
            
            # Convert to distance (0 = identical, 1 = completely different)
            cosine_distance = 1 - cosine_sim
            
            return float(cosine_distance)
            
        except Exception as e:
            logger.error(f"❌ Cosine distance calculation error: {e}")
            return 1.0  # Return max distance on error
    
    def _create_unknown_result(self) -> Dict:
        """Create result for unknown/unrecognized face"""
        return {
            'type': 'unknown',
            'employee': None,
            'confidence': 0.0,
            'match_method': 'none'
        }
    
    def save_image_and_vector(self, file, employee_id: int):
        """Save uploaded image and extract face vector"""
        try:
            if not self.db_manager:
                raise Exception("Database manager not available")
            
            # Create upload directory if not exists
            upload_dir = 'static/uploads/faces'
            os.makedirs(upload_dir, exist_ok=True)
            
            # Generate unique filename
            file_extension = os.path.splitext(file.filename)[1].lower()
            unique_filename = f"{employee_id}_{uuid.uuid4().hex}{file_extension}"
            file_path = os.path.join(upload_dir, unique_filename)
            
            # Save image file
            if hasattr(file, 'save'):
                file.save(file_path)
            else:
                # Handle file-like objects (from laptop capture)
                with open(file_path, 'wb') as f:
                    file.seek(0)
                    f.write(file.read())
            
            # Load and process image
            image = cv2.imread(file_path)
            if image is None:
                raise Exception("Could not load saved image")
            
            # Extract face vector using DeepFace
            if DEEPFACE_AVAILABLE:
                face_embedding = self._extract_face_embedding(image)
                if face_embedding is None:
                    raise Exception("Could not extract face embedding from image")
            else:
                # Fallback: create dummy vector
                face_embedding = np.random.rand(512)  # Facenet512 size
            
            # Save to database
            session = self.db_manager.get_session()
            
            try:
                vector_face = VectorFace(
                    employee_id=employee_id,
                    vector_data=json.dumps(face_embedding.tolist()),
                    image_path=file_path
                )
                
                session.add(vector_face)
                session.commit()
                session.refresh(vector_face)
                
                # Update in-memory database
                self._update_face_database(employee_id, vector_face, face_embedding)
                
                logger.info(f"✅ Saved face vector for employee {employee_id}")
                return vector_face
                
            except Exception as e:
                session.rollback()
                # Clean up file on database error
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise e
                
            finally:
                self.db_manager.close_session(session)
                
        except Exception as e:
            logger.error(f"❌ Error saving image and vector: {e}")
            raise e
    
    def _update_face_database(self, employee_id: int, vector_face, face_embedding: np.ndarray):
        """Update in-memory face database with new vector"""
        try:
            if employee_id not in self.face_db:
                # Get employee info
                session = self.db_manager.get_session()
                employee = session.query(Employee).filter_by(id=employee_id).first()
                
                if employee:
                    self.face_db[employee_id] = {
                        'employee': {
                            'id': employee.id,
                            'name': employee.name,
                            'employee_code': employee.employee_code,
                            'department': employee.department,
                            'position': employee.position
                        },
                        'vectors': []
                    }
                
                self.db_manager.close_session(session)
            
            # Add new vector
            if employee_id in self.face_db:
                self.face_db[employee_id]['vectors'].append({
                    'id': vector_face.id,
                    'vector': face_embedding,
                    'image_path': vector_face.image_path
                })
                
        except Exception as e:
            logger.error(f"❌ Error updating face database: {e}")
    
    def get_employee_vectors(self, employee_id: int) -> List:
        """Get all face vectors for an employee"""
        try:
            if not self.db_manager:
                return []
            
            session = self.db_manager.get_session()
            vectors = session.query(VectorFace).filter_by(employee_id=employee_id).all()
            self.db_manager.close_session(session)
            
            return vectors
            
        except Exception as e:
            logger.error(f"❌ Error getting employee vectors: {e}")
            return []
    
    def get_employee_image_count(self, employee_id: int) -> int:
        """Get number of face images for employee"""
        try:
            if not self.db_manager:
                return 0
            
            session = self.db_manager.get_session()
            count = session.query(VectorFace).filter_by(employee_id=employee_id).count()
            self.db_manager.close_session(session)
            
            return count
            
        except Exception as e:
            logger.error(f"❌ Error getting image count: {e}")
            return 0
    
    def delete_face_vector(self, vector_id: int) -> bool:
        """Delete a specific face vector"""
        try:
            if not self.db_manager:
                return False
            
            session = self.db_manager.get_session()
            
            vector = session.query(VectorFace).filter_by(id=vector_id).first()
            if vector:
                # Delete image file
                if os.path.exists(vector.image_path):
                    os.remove(vector.image_path)
                
                # Remove from database
                session.delete(vector)
                session.commit()
                
                # Remove from in-memory database
                self._remove_from_face_database(vector.employee_id, vector_id)
                
                self.db_manager.close_session(session)
                return True
            else:
                self.db_manager.close_session(session)
                return False
                
        except Exception as e:
            logger.error(f"❌ Error deleting face vector: {e}")
            return False
    
    def delete_all_employee_vectors(self, employee_id: int) -> bool:
        """Delete all face vectors for an employee"""
        try:
            if not self.db_manager:
                return False
            
            session = self.db_manager.get_session()
            
            vectors = session.query(VectorFace).filter_by(employee_id=employee_id).all()
            
            for vector in vectors:
                # Delete image file
                if os.path.exists(vector.image_path):
                    os.remove(vector.image_path)
                
                session.delete(vector)
            
            session.commit()
            
            # Remove from in-memory database
            if employee_id in self.face_db:
                del self.face_db[employee_id]
            
            self.db_manager.close_session(session)
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting all employee vectors: {e}")
            return False
    
    def _remove_from_face_database(self, employee_id: int, vector_id: int):
        """Remove specific vector from in-memory database"""
        try:
            if employee_id in self.face_db:
                self.face_db[employee_id]['vectors'] = [
                    v for v in self.face_db[employee_id]['vectors'] 
                    if v['id'] != vector_id
                ]
                
                # Remove employee if no vectors left
                if not self.face_db[employee_id]['vectors']:
                    del self.face_db[employee_id]
                    
        except Exception as e:
            logger.error(f"❌ Error removing from face database: {e}")
    
    def update_threshold(self, new_threshold: float):
        """Update recognition threshold"""
        if 0.0 <= new_threshold <= 1.0:
            self.threshold = new_threshold
            logger.info(f"🔧 Updated recognition threshold to {new_threshold}")
    
    def get_recognition_stats(self) -> Dict:
        """Get face recognition statistics"""
        avg_processing_time = (
            sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            if self.stats['processing_times'] else 0
        )
        
        success_rate = (
            self.stats['successful_matches'] / self.stats['total_recognitions']
            if self.stats['total_recognitions'] > 0 else 0
        )
        
        return {
            'total_recognitions': self.stats['total_recognitions'],
            'successful_matches': self.stats['successful_matches'],
            'success_rate': round(success_rate * 100, 2),
            'average_processing_time_ms': round(avg_processing_time * 1000, 2),
            'face_database_size': len(self.face_db),
            'total_vectors': sum(len(emp['vectors']) for emp in self.face_db.values()),
            'threshold': self.threshold,
            'model_name': self.model_name,
            'deepface_available': DEEPFACE_AVAILABLE
        }
    
    def reload_face_database(self):
        """Reload face database from disk (after updates)"""
        logger.info("🔄 Reloading face database...")
        self.face_db = {}
        self._load_face_database()
    
    def test_recognition_pipeline(self, test_image_path: str) -> Dict:
        """Test the recognition pipeline with a test image"""
        try:
            # Load test image
            test_image = cv2.imread(test_image_path)
            if test_image is None:
                return {'success': False, 'error': 'Could not load test image'}
            
            # Run recognition
            start_time = time.time()
            results = self.recognize_face(test_image)
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'processing_time_ms': round(processing_time * 1000, 2),
                'faces_detected': len(results),
                'results': results,
                'image_size': test_image.shape,
                'model_info': {
                    'model_name': self.model_name,
                    'threshold': self.threshold,
                    'deepface_available': DEEPFACE_AVAILABLE
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}