#!/usr/bin/env python3
"""
face_service.py - FACE RECOGNITION SERVICE
Advanced face recognition with DeepFace integration
Real-time video processing and face vector matching
"""

import cv2
import numpy as np
import os
import logging
import time
import hashlib
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import base64
from PIL import Image
import io

# DeepFace imports with fallback
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not available - using basic face detection")

# Database imports
from ..models.models import VectorFace, Employee, DatabaseManager
from ..database.database import db_session_scope
from ..config import Config

logger = logging.getLogger(__name__)


class FaceService:
    """Advanced face recognition service with DeepFace integration"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager or DatabaseManager()
        self.config = Config()
        
        # DeepFace configuration
        self.model_name = self.config.DEEPFACE_MODEL_NAME
        self.detector_backend = self.config.DEEPFACE_DETECTOR_BACKEND
        self.distance_metric = self.config.DEEPFACE_DISTANCE_METRIC
        self.enforce_detection = self.config.DEEPFACE_ENFORCE_DETECTION
        self.align = self.config.DEEPFACE_ALIGN
        
        # Face detection fallback
        self.face_cascade = None
        self._init_opencv_fallback()
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'average_processing_time': 0.0,
            'last_processed': None
        }
        
        logger.info(f"✅ FaceService initialized - DeepFace: {DEEPFACE_AVAILABLE}")
    
    def _init_opencv_fallback(self):
        """Initialize OpenCV cascade for fallback face detection"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("✅ OpenCV face cascade initialized")
            else:
                logger.warning("⚠️ OpenCV face cascade not found")
        except Exception as e:
            logger.error(f"❌ Error initializing OpenCV cascade: {e}")
    
    def extract_face_embedding(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding using DeepFace"""
        if not DEEPFACE_AVAILABLE:
            logger.warning("DeepFace not available - cannot extract embeddings")
            return None
        
        try:
            start_time = time.time()
            
            # Convert BGR to RGB if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_array
            
            # Extract embedding using DeepFace
            embedding = DeepFace.represent(
                img_path=image_rgb,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                align=self.align
            )
            
            processing_time = time.time() - start_time
            logger.debug(f"Face embedding extracted in {processing_time:.3f}s")
            
            # DeepFace returns a list of dictionaries
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]["embedding"])
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting face embedding: {e}")
            return None
    
    def detect_faces_opencv(self, image_array: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Fallback face detection using OpenCV"""
        if self.face_cascade is None:
            return []
        
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.config.FACE_DETECTION_SCALE_FACTOR,
                minNeighbors=self.config.FACE_MIN_NEIGHBORS,
                minSize=(30, 30)
            )
            
            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
            
        except Exception as e:
            logger.error(f"❌ Error in OpenCV face detection: {e}")
            return []
    
    def detect_and_recognize_faces(self, image_array: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and recognize faces in image"""
        start_time = time.time()
        results = []
        
        try:
            if DEEPFACE_AVAILABLE:
                results = self._deepface_detection_and_recognition(image_array)
            else:
                results = self._opencv_fallback_detection(image_array)
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats['total_processed'] += 1
            self.stats['last_processed'] = datetime.now()
            
            # Update average processing time
            if self.stats['average_processing_time'] == 0:
                self.stats['average_processing_time'] = processing_time
            else:
                self.stats['average_processing_time'] = (
                    self.stats['average_processing_time'] * 0.9 + processing_time * 0.1
                )
            
            if results:
                self.stats['successful_recognitions'] += len([r for r in results if r.get('employee')])
            else:
                self.stats['failed_recognitions'] += 1
            
            logger.debug(f"Face detection completed in {processing_time:.3f}s - {len(results)} faces found")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in face detection and recognition: {e}")
            self.stats['failed_recognitions'] += 1
            return []
    
    def _deepface_detection_and_recognition(self, image_array: np.ndarray) -> List[Dict[str, Any]]:
        """Advanced detection and recognition using DeepFace"""
        results = []
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # First, detect faces and get their locations
            face_objs = DeepFace.extract_faces(
                img_path=image_rgb,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=self.align
            )
            
            if not face_objs:
                return []
            
            # Process each detected face
            for i, face_obj in enumerate(face_objs):
                try:
                    # Extract embedding for this face
                    face_embedding = DeepFace.represent(
                        img_path=face_obj,
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        enforce_detection=False,
                        align=self.align
                    )
                    
                    if face_embedding and len(face_embedding) > 0:
                        embedding_vector = np.array(face_embedding[0]["embedding"])
                        
                        # Find matching employee
                        matched_employee = self._find_matching_employee(embedding_vector)
                        
                        # Calculate bounding box (approximation for DeepFace)
                        h, w = image_array.shape[:2]
                        bbox = self._estimate_bbox_from_face_obj(face_obj, w, h, i)
                        
                        result = {
                            'type': 'recognized' if matched_employee else 'unknown',
                            'employee': matched_employee,
                            'name': matched_employee['name'] if matched_employee else 'Unknown Person',
                            'confidence': matched_employee['confidence'] if matched_employee else 0.0,
                            'bbox': bbox,
                            'embedding': embedding_vector,
                            'detection_method': 'deepface'
                        }
                        
                        results.append(result)
                        
                except Exception as e:
                    logger.error(f"❌ Error processing face {i}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in DeepFace detection: {e}")
            return self._opencv_fallback_detection(image_array)
    
    def _opencv_fallback_detection(self, image_array: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback detection using OpenCV"""
        results = []
        faces = self.detect_faces_opencv(image_array)
        
        for (x, y, w, h) in faces:
            result = {
                'type': 'unknown',
                'employee': None,
                'name': 'Unknown Person',
                'confidence': 0.75,  # Default confidence for OpenCV detection
                'bbox': [x, y, w, h],
                'embedding': None,
                'detection_method': 'opencv'
            }
            results.append(result)
        
        return results
    
    def _estimate_bbox_from_face_obj(self, face_obj, img_w: int, img_h: int, face_index: int) -> List[int]:
        """Estimate bounding box from DeepFace face object"""
        # This is an approximation since DeepFace doesn't always provide exact coordinates
        # We'll use the face dimensions and make reasonable estimates
        
        try:
            if hasattr(face_obj, 'shape'):
                face_h, face_w = face_obj.shape[:2]
            else:
                # Default face size estimation
                face_w, face_h = 150, 150
            
            # Estimate position (this is rough - DeepFace doesn't give exact coordinates)
            center_x = img_w // 2
            center_y = img_h // 2
            
            # Adjust for multiple faces
            if face_index > 0:
                center_x += (face_index - 0.5) * face_w
            
            x = max(0, center_x - face_w // 2)
            y = max(0, center_y - face_h // 2)
            
            # Ensure bbox stays within image bounds
            x = min(x, img_w - face_w)
            y = min(y, img_h - face_h)
            
            return [int(x), int(y), int(face_w), int(face_h)]
            
        except Exception as e:
            logger.error(f"❌ Error estimating bbox: {e}")
            # Return a default centered bbox
            return [img_w//4, img_h//4, img_w//2, img_h//2]
    
    def _find_matching_employee(self, embedding_vector: np.ndarray) -> Optional[Dict[str, Any]]:
        """Find matching employee by comparing face embeddings"""
        try:
            with db_session_scope() as session:
                # Get all face vectors from database
                face_vectors = session.query(VectorFace).join(Employee).filter(
                    Employee.is_active == True
                ).all()
                
                if not face_vectors:
                    return None
                
                best_match = None
                best_distance = float('inf')
                
                for face_vector in face_vectors:
                    try:
                        stored_embedding = face_vector.vector_array
                        if stored_embedding is None:
                            continue
                        
                        # Calculate distance using configured metric
                        if self.distance_metric == 'cosine':
                            distance = self._cosine_distance(embedding_vector, stored_embedding)
                        elif self.distance_metric == 'euclidean':
                            distance = self._euclidean_distance(embedding_vector, stored_embedding)
                        else:
                            distance = self._euclidean_l2_distance(embedding_vector, stored_embedding)
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_match = {
                                'employee_id': face_vector.employee.id,
                                'name': face_vector.employee.name,
                                'employee_code': face_vector.employee.employee_code,
                                'department': face_vector.employee.department,
                                'distance': distance,
                                'confidence': max(0.0, 1.0 - distance),  # Convert distance to confidence
                                'vector_id': face_vector.id
                            }
                    
                    except Exception as e:
                        logger.error(f"❌ Error comparing with vector {face_vector.id}: {e}")
                        continue
                
                # Check if best match meets threshold
                if best_match and best_distance <= self.config.FACE_RECOGNITION_THRESHOLD:
                    return best_match
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Error finding matching employee: {e}")
            return None
    
    def _cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine distance between two vectors"""
        try:
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 1.0
            
            cosine_similarity = dot_product / (norm_v1 * norm_v2)
            return 1.0 - cosine_similarity
            
        except Exception as e:
            logger.error(f"❌ Error calculating cosine distance: {e}")
            return 1.0
    
    def _euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate Euclidean distance between two vectors"""
        try:
            return np.linalg.norm(v1 - v2)
        except Exception as e:
            logger.error(f"❌ Error calculating Euclidean distance: {e}")
            return float('inf')
    
    def _euclidean_l2_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate L2 normalized Euclidean distance"""
        try:
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            return np.linalg.norm(v1_norm - v2_norm)
        except Exception as e:
            logger.error(f"❌ Error calculating L2 distance: {e}")
            return float('inf')
    
    def save_image_and_vector(self, file, employee_id: int) -> VectorFace:
        """Save uploaded image and extract face vector"""
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            filename = f"emp_{employee_id}_{timestamp}_{file_hash}.jpg"
            
            # Ensure upload directory exists
            upload_dir = self.config.UPLOAD_FOLDER
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, filename)
            
            # Read and process image
            file.seek(0)  # Reset file pointer
            image_data = file.read()
            
            # Convert to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image_array is None:
                raise ValueError("Could not decode image")
            
            # Resize if too large
            h, w = image_array.shape[:2]
            max_size = self.config.FACE_MAX_SIZE
            if w > max_size[0] or h > max_size[1]:
                scale = min(max_size[0]/w, max_size[1]/h)
                new_w, new_h = int(w*scale), int(h*scale)
                image_array = cv2.resize(image_array, (new_w, new_h))
            
            # Save processed image
            cv2.imwrite(file_path, image_array)
            
            # Extract face embedding
            embedding = self.extract_face_embedding(image_array)
            
            if embedding is None:
                # If DeepFace fails, still save the image but with null embedding
                logger.warning(f"Could not extract face embedding for employee {employee_id}")
            
            # Save to database
            with db_session_scope() as session:
                vector_face = VectorFace(
                    employee_id=employee_id,
                    image_path=file_path,
                    model_name=self.model_name,
                    detection_backend=self.detector_backend,
                    confidence_score=1.0 if embedding is not None else 0.0
                )
                
                if embedding is not None:
                    vector_face.vector_array = embedding
                
                session.add(vector_face)
                session.commit()
                
                # Refresh to get the ID
                session.refresh(vector_face)
                
                logger.info(f"✅ Saved face vector for employee {employee_id}: {vector_face.id}")
                return vector_face
        
        except Exception as e:
            logger.error(f"❌ Error saving image and vector: {e}")
            # Clean up file if it was created
            if 'file_path' in locals() and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            raise
    
    def get_employee_image_count(self, employee_id: int) -> int:
        """Get number of face images for employee"""
        try:
            with db_session_scope() as session:
                count = session.query(VectorFace).filter(
                    VectorFace.employee_id == employee_id
                ).count()
                return count
        except Exception as e:
            logger.error(f"❌ Error getting image count: {e}")
            return 0
    
    def get_employee_vectors(self, employee_id: int) -> List[VectorFace]:
        """Get all face vectors for employee"""
        try:
            with db_session_scope() as session:
                vectors = session.query(VectorFace).filter(
                    VectorFace.employee_id == employee_id
                ).order_by(VectorFace.created_at.desc()).all()
                
                # Return detached objects
                return [
                    type('VectorFace', (), {
                        'id': v.id,
                        'image_path': v.image_path,
                        'created_at': v.created_at,
                        'model_name': v.model_name,
                        'confidence_score': v.confidence_score
                    })() for v in vectors
                ]
        except Exception as e:
            logger.error(f"❌ Error getting employee vectors: {e}")
            return []
    
    def delete_face_vector(self, vector_id: int) -> bool:
        """Delete specific face vector"""
        try:
            with db_session_scope() as session:
                vector = session.query(VectorFace).filter(
                    VectorFace.id == vector_id
                ).first()
                
                if not vector:
                    return False
                
                # Delete image file
                if vector.image_path and os.path.exists(vector.image_path):
                    try:
                        os.remove(vector.image_path)
                    except Exception as e:
                        logger.warning(f"Could not delete image file: {e}")
                
                # Delete from database
                session.delete(vector)
                session.commit()
                
                logger.info(f"✅ Deleted face vector {vector_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error deleting face vector: {e}")
            return False
    
    def delete_all_employee_vectors(self, employee_id: int) -> bool:
        """Delete all face vectors for employee"""
        try:
            with db_session_scope() as session:
                vectors = session.query(VectorFace).filter(
                    VectorFace.employee_id == employee_id
                ).all()
                
                # Delete image files
                for vector in vectors:
                    if vector.image_path and os.path.exists(vector.image_path):
                        try:
                            os.remove(vector.image_path)
                        except Exception as e:
                            logger.warning(f"Could not delete image file: {e}")
                
                # Delete from database
                deleted_count = session.query(VectorFace).filter(
                    VectorFace.employee_id == employee_id
                ).delete()
                
                session.commit()
                
                logger.info(f"✅ Deleted {deleted_count} face vectors for employee {employee_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error deleting employee vectors: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get face recognition service statistics"""
        return {
            'deepface_available': DEEPFACE_AVAILABLE,
            'model_name': self.model_name,
            'detector_backend': self.detector_backend,
            'distance_metric': self.distance_metric,
            'recognition_threshold': self.config.FACE_RECOGNITION_THRESHOLD,
            'processing_stats': self.stats.copy(),
            'opencv_available': self.face_cascade is not None
        }
    
    def test_service(self) -> Dict[str, Any]:
        """Test face recognition service"""
        try:
            # Create test image
            test_image = np.zeros((200, 200, 3), dtype=np.uint8)
            test_image.fill(128)  # Gray image
            
            # Test detection
            start_time = time.time()
            results = self.detect_and_recognize_faces(test_image)
            processing_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'deepface_available': DEEPFACE_AVAILABLE,
                'opencv_available': self.face_cascade is not None,
                'test_processing_time': processing_time,
                'test_results_count': len(results),
                'stats': self.get_stats()
            }
            
        except Exception as e:
            logger.error(f"❌ Face service test failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'deepface_available': DEEPFACE_AVAILABLE,
                'opencv_available': self.face_cascade is not None
            }