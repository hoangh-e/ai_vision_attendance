#!/usr/bin/env python3
"""
enhanced_face_service.py - DEEPFACE INTEGRATION FOR COMMERCIAL EMPLOYEE RECOGNITION
File: backend/services/enhanced_face_service.py
"""

import cv2
import numpy as np
import base64
import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import uuid
from pathlib import Path

# DeepFace imports
try:
    from deepface import DeepFace
    from deepface.commons import functions as DeepFaceUtils
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# Database imports
from database.models import Employee, VectorFace, AttendanceLog
from database.database import DatabaseManager

logger = logging.getLogger(__name__)

class EnhancedFaceService:
    """
    Commercial-grade face recognition service using DeepFace
    Features:
    - Employee image management with vector extraction
    - Real-time face recognition against database
    - High accuracy similarity matching
    - Scalable architecture for enterprise use
    """
    
    def __init__(self, 
                 model_name: str = 'Facenet512',
                 detector_backend: str = 'opencv',
                 similarity_threshold: float = 0.6,
                 max_images_per_employee: int = 10):
        """
        Initialize Enhanced Face Service with DeepFace
        
        Args:
            model_name: DeepFace model ('Facenet512', 'VGG-Face', 'ArcFace')
            detector_backend: Face detection backend ('opencv', 'mtcnn', 'ssd')
            similarity_threshold: Similarity threshold for recognition (0.0-1.0)
            max_images_per_employee: Maximum images per employee
        """
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.similarity_threshold = similarity_threshold
        self.max_images_per_employee = max_images_per_employee
        
        # Database manager
        self.db_manager = DatabaseManager()
        
        # Face detection cascade (backup)
        self.face_cascade = None
        
        # Performance stats
        self.stats = {
            'total_recognitions': 0,
            'successful_matches': 0,
            'processing_times': []
        }
        
        # Initialize components
        self.initialize_deepface()
        self.initialize_opencv_backup()
        
        logger.info(f"✅ Enhanced Face Service initialized:")
        logger.info(f"   - Model: {self.model_name}")
        logger.info(f"   - Detector: {self.detector_backend}")
        logger.info(f"   - Threshold: {self.similarity_threshold}")
        logger.info(f"   - DeepFace: {'Available' if DEEPFACE_AVAILABLE else 'Not Available'}")

    def initialize_deepface(self):
        """Initialize DeepFace components"""
        if not DEEPFACE_AVAILABLE:
            logger.warning("⚠️ DeepFace not available. Install: pip install deepface")
            return False
        
        try:
            # Test DeepFace functionality
            logger.info("🔄 Initializing DeepFace...")
            
            # Create a dummy image for testing
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            dummy_path = "temp_test_image.jpg"
            cv2.imwrite(dummy_path, dummy_img)
            
            try:
                # Test face detection
                faces = DeepFace.extract_faces(
                    img_path=dummy_path,
                    detector_backend=self.detector_backend,
                    enforce_detection=False
                )
                logger.info(f"✅ DeepFace face detection working")
                
                # Test embedding extraction
                if len(faces) > 0:
                    embedding = DeepFace.represent(
                        img_path=dummy_path,
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        enforce_detection=False
                    )
                    logger.info(f"✅ DeepFace embedding extraction working")
                    logger.info(f"   - Embedding dimension: {len(embedding[0]['embedding'])}")
                
            finally:
                # Clean up test file
                if os.path.exists(dummy_path):
                    os.remove(dummy_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ DeepFace initialization failed: {e}")
            return False

    def initialize_opencv_backup(self):
        """Initialize OpenCV as backup for face detection"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                self.face_cascade = None
                logger.warning("⚠️ OpenCV face cascade not loaded")
            else:
                logger.info("✅ OpenCV face cascade loaded as backup")
                
        except Exception as e:
            logger.error(f"❌ OpenCV backup initialization failed: {e}")
            self.face_cascade = None

    def save_employee_image_with_vector(self, image_file, employee_id: int) -> Dict[str, Any]:
        """
        Save employee image and extract face vector using DeepFace
        
        Args:
            image_file: Uploaded image file
            employee_id: Employee ID
            
        Returns:
            Dict with success status and vector info
        """
        session = self.db_manager.get_session()
        
        try:
            # Check employee exists
            employee = session.query(Employee).filter_by(id=employee_id).first()
            if not employee:
                return {'success': False, 'error': 'Employee not found'}
            
            # Check image limit
            current_count = session.query(VectorFace).filter_by(employee_id=employee_id).count()
            if current_count >= self.max_images_per_employee:
                return {
                    'success': False, 
                    'error': f'Employee already has maximum {self.max_images_per_employee} images'
                }
            
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            file_extension = os.path.splitext(image_file.filename)[1].lower()
            filename = f"emp_{employee_id}_{timestamp}_{unique_id}{file_extension}"
            
            # Create upload directory
            upload_dir = Path("../frontend/static/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Save image file
            image_path = upload_dir / filename
            image_file.save(str(image_path))
            
            logger.info(f"📁 Saved image: {filename}")
            
            # Extract face vector using DeepFace
            vector_result = self.extract_face_vector_from_file(str(image_path))
            
            if not vector_result['success']:
                # Clean up file if vector extraction failed
                if image_path.exists():
                    image_path.unlink()
                return vector_result
            
            # Save vector to database
            vector_face = VectorFace(
                employee_id=employee_id,
                vector_data=json.dumps(vector_result['vector']),
                image_path=f"static/uploads/{filename}",
                created_at=datetime.utcnow()
            )
            
            session.add(vector_face)
            session.commit()
            session.refresh(vector_face)
            
            logger.info(f"✅ Saved face vector for employee {employee_id}")
            
            return {
                'success': True,
                'vector_id': vector_face.id,
                'image_path': vector_face.image_path,
                'employee_name': employee.name,
                'vector_dimension': len(vector_result['vector']),
                'confidence': vector_result.get('confidence', 0.0),
                'total_images': current_count + 1
            }
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error saving employee image with vector: {e}")
            return {'success': False, 'error': str(e)}
            
        finally:
            self.db_manager.close_session(session)

    def extract_face_vector_from_file(self, image_path: str) -> Dict[str, Any]:
        """
        Extract face vector from image file using DeepFace
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with vector data and metadata
        """
        if not DEEPFACE_AVAILABLE:
            return {'success': False, 'error': 'DeepFace not available'}
        
        try:
            # Verify image exists
            if not os.path.exists(image_path):
                return {'success': False, 'error': 'Image file not found'}
            
            # Extract faces first
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector_backend,
                enforce_detection=True  # Require face detection
            )
            
            if len(faces) == 0:
                return {'success': False, 'error': 'No face detected in image'}
            
            if len(faces) > 1:
                logger.warning(f"⚠️ Multiple faces detected, using largest face")
            
            # Extract embedding/vector
            embeddings = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True
            )
            
            if len(embeddings) == 0:
                return {'success': False, 'error': 'Failed to extract face vector'}
            
            # Get the first (or largest) face embedding
            primary_embedding = embeddings[0]
            vector = primary_embedding['embedding']
            
            logger.info(f"✅ Extracted face vector: {len(vector)} dimensions")
            
            return {
                'success': True,
                'vector': vector,
                'confidence': 1.0,  # DeepFace doesn't provide confidence for extraction
                'faces_detected': len(faces),
                'vector_dimension': len(vector),
                'model_used': self.model_name
            }
            
        except Exception as e:
            logger.error(f"❌ Face vector extraction failed: {e}")
            return {'success': False, 'error': str(e)}

    def recognize_faces_in_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Recognize faces in video frame against employee database
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            List of recognition results with employee info
        """
        start_time = datetime.now()
        
        try:
            if not DEEPFACE_AVAILABLE:
                # Fallback to basic detection
                return self.fallback_face_detection(frame)
            
            # Save frame temporarily for DeepFace processing
            temp_frame_path = f"temp_frame_{int(datetime.now().timestamp())}.jpg"
            cv2.imwrite(temp_frame_path, frame)
            
            try:
                # Extract faces from frame
                faces = DeepFace.extract_faces(
                    img_path=temp_frame_path,
                    detector_backend=self.detector_backend,
                    enforce_detection=False
                )
                
                if len(faces) == 0:
                    return []
                
                # Get face regions for bounding boxes
                face_objs = DeepFace.extract_faces(
                    img_path=temp_frame_path,
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    align=False  # Keep original coordinates
                )
                
                # Extract embeddings
                embeddings = DeepFace.represent(
                    img_path=temp_frame_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=False
                )
                
                results = []
                
                # Process each detected face
                for i, (face, embedding_data) in enumerate(zip(faces, embeddings)):
                    
                    # Get face vector
                    face_vector = embedding_data['embedding']
                    
                    # Find matching employee
                    match_result = self.find_matching_employee(face_vector)
                    
                    # Calculate bounding box (approximate)
                    height, width = frame.shape[:2]
                    face_region = face_objs[i] if i < len(face_objs) else None
                    
                    # Default bounding box if region not available
                    bbox = self.estimate_face_bbox(frame, i, len(faces))
                    
                    # Create result
                    result = {
                        'bbox': bbox,
                        'confidence': match_result.get('similarity', 0.0),
                        'employee': match_result.get('employee'),
                        'match_found': match_result['success'],
                        'processing_time': (datetime.now() - start_time).total_seconds(),
                        'vector_dimension': len(face_vector)
                    }
                    
                    results.append(result)
                    
                    # Update stats
                    self.stats['total_recognitions'] += 1
                    if match_result['success']:
                        self.stats['successful_matches'] += 1
                
                # Log attendance if employees recognized
                self.log_attendance(results)
                
                return results
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
                    
        except Exception as e:
            logger.error(f"❌ Face recognition error: {e}")
            return self.fallback_face_detection(frame)
        
        finally:
            # Update performance stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['processing_times'].append(processing_time)
            # Keep only last 100 processing times
            if len(self.stats['processing_times']) > 100:
                self.stats['processing_times'] = self.stats['processing_times'][-100:]

    def find_matching_employee(self, query_vector: List[float]) -> Dict[str, Any]:
        """
        Find matching employee by comparing face vector against database
        
        Args:
            query_vector: Face vector to match
            
        Returns:
            Dict with matching result and employee info
        """
        session = self.db_manager.get_session()
        
        try:
            # Get all employee vectors from database
            all_vectors = session.query(VectorFace).join(Employee).all()
            
            if len(all_vectors) == 0:
                return {'success': False, 'reason': 'No employee vectors in database'}
            
            best_match = None
            best_similarity = 0.0
            best_employee = None
            
            # Compare with each stored vector
            for vector_record in all_vectors:
                try:
                    stored_vector = json.loads(vector_record.vector_data)
                    
                    # Calculate similarity (cosine similarity)
                    similarity = self.calculate_vector_similarity(query_vector, stored_vector)
                    
                    # Check if this is the best match so far
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = vector_record
                        best_employee = vector_record.employee
                
                except Exception as e:
                    logger.warning(f"⚠️ Error processing vector {vector_record.id}: {e}")
                    continue
            
            if best_match and best_employee:
                logger.info(f"✅ Employee recognized: {best_employee.name} (similarity: {best_similarity:.3f})")
                
                return {
                    'success': True,
                    'employee': {
                        'id': best_employee.id,
                        'name': best_employee.name,
                        'employee_code': best_employee.employee_code,
                        'department': best_employee.department,
                        'position': best_employee.position
                    },
                    'similarity': best_similarity,
                    'vector_id': best_match.id,
                    'confidence': best_similarity
                }
            else:
                logger.info(f"❌ No employee match found (best similarity: {best_similarity:.3f}, threshold: {self.similarity_threshold})")
                return {
                    'success': False, 
                    'reason': 'No matching employee found',
                    'best_similarity': best_similarity,
                    'threshold': self.similarity_threshold
                }
                
        except Exception as e:
            logger.error(f"❌ Error finding matching employee: {e}")
            return {'success': False, 'error': str(e)}
            
        finally:
            self.db_manager.close_session(session)

    def calculate_vector_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate cosine similarity between two face vectors
        
        Args:
            vector1: First face vector
            vector2: Second face vector
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Convert to numpy arrays
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            similarity = (similarity + 1) / 2
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"❌ Similarity calculation error: {e}")
            return 0.0

    def estimate_face_bbox(self, frame: np.ndarray, face_index: int, total_faces: int) -> List[int]:
        """
        Estimate face bounding box when precise coordinates unavailable
        
        Args:
            frame: Video frame
            face_index: Index of the face
            total_faces: Total number of faces detected
            
        Returns:
            Bounding box [x, y, width, height]
        """
        height, width = frame.shape[:2]
        
        # Use OpenCV as backup for bounding box
        if self.face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                
                if len(faces) > face_index:
                    x, y, w, h = faces[face_index]
                    return [int(x), int(y), int(w), int(h)]
                    
            except Exception as e:
                logger.debug(f"OpenCV backup bbox failed: {e}")
        
        # Fallback: estimate bbox based on frame size
        estimated_size = min(width, height) // 4
        x = width // 4 + (face_index * width // (total_faces + 1))
        y = height // 4
        
        return [x, y, estimated_size, estimated_size]

    def fallback_face_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Fallback face detection using OpenCV when DeepFace unavailable
        
        Args:
            frame: Video frame
            
        Returns:
            List of detection results
        """
        if self.face_cascade is None:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            results = []
            for (x, y, w, h) in faces:
                results.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': 0.8,  # Default confidence
                    'employee': None,   # No recognition without DeepFace
                    'match_found': False,
                    'processing_time': 0.01,
                    'fallback_detection': True
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Fallback detection error: {e}")
            return []

    def log_attendance(self, recognition_results: List[Dict[str, Any]]):
        """
        Log attendance for recognized employees
        
        Args:
            recognition_results: List of recognition results
        """
        session = self.db_manager.get_session()
        
        try:
            for result in recognition_results:
                if result.get('match_found') and result.get('employee'):
                    employee_id = result['employee']['id']
                    confidence = result.get('confidence', 0.0)
                    
                    # Check if already logged today
                    today = datetime.utcnow().date()
                    existing_log = session.query(AttendanceLog).filter(
                        AttendanceLog.employee_id == employee_id,
                        AttendanceLog.check_in_time >= today
                    ).first()
                    
                    if not existing_log:
                        # Create new attendance log
                        attendance_log = AttendanceLog(
                            employee_id=employee_id,
                            check_in_time=datetime.utcnow(),
                            confidence_score=confidence
                        )
                        
                        session.add(attendance_log)
                        session.commit()
                        
                        logger.info(f"📝 Logged attendance: {result['employee']['name']}")
                        
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Attendance logging error: {e}")
            
        finally:
            self.db_manager.close_session(session)

    def get_employee_images(self, employee_id: int) -> List[Dict[str, Any]]:
        """
        Get all images for an employee
        
        Args:
            employee_id: Employee ID
            
        Returns:
            List of image information
        """
        session = self.db_manager.get_session()
        
        try:
            vectors = session.query(VectorFace).filter_by(employee_id=employee_id).all()
            
            results = []
            for vector in vectors:
                results.append({
                    'id': vector.id,
                    'image_path': vector.image_path,
                    'created_at': vector.created_at.isoformat() if vector.created_at else None,
                    'has_vector': bool(vector.vector_data)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error getting employee images: {e}")
            return []
            
        finally:
            self.db_manager.close_session(session)

    def delete_employee_image(self, vector_id: int) -> bool:
        """
        Delete an employee image and its vector
        
        Args:
            vector_id: Vector face ID
            
        Returns:
            Success status
        """
        session = self.db_manager.get_session()
        
        try:
            vector = session.query(VectorFace).filter_by(id=vector_id).first()
            if not vector:
                return False
            
            # Delete image file
            if vector.image_path:
                image_path = Path("../frontend") / vector.image_path
                if image_path.exists():
                    image_path.unlink()
                    logger.info(f"🗑️ Deleted image file: {vector.image_path}")
            
            # Delete database record
            session.delete(vector)
            session.commit()
            
            logger.info(f"✅ Deleted employee image vector: {vector_id}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error deleting employee image: {e}")
            return False
            
        finally:
            self.db_manager.close_session(session)

    def get_recognition_stats(self) -> Dict[str, Any]:
        """
        Get recognition performance statistics
        
        Returns:
            Performance stats dictionary
        """
        avg_processing_time = 0.0
        if self.stats['processing_times']:
            avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
        
        success_rate = 0.0
        if self.stats['total_recognitions'] > 0:
            success_rate = self.stats['successful_matches'] / self.stats['total_recognitions']
        
        return {
            'total_recognitions': self.stats['total_recognitions'],
            'successful_matches': self.stats['successful_matches'],
            'success_rate': success_rate,
            'average_processing_time': avg_processing_time,
            'model_name': self.model_name,
            'similarity_threshold': self.similarity_threshold,
            'deepface_available': DEEPFACE_AVAILABLE
        }

    def update_threshold(self, new_threshold: float):
        """Update similarity threshold"""
        if 0.0 <= new_threshold <= 1.0:
            self.similarity_threshold = new_threshold
            logger.info(f"✅ Updated similarity threshold to: {new_threshold}")

    def get_employee_image_count(self, employee_id: int) -> int:
        """Get number of images for an employee"""
        session = self.db_manager.get_session()
        try:
            count = session.query(VectorFace).filter_by(employee_id=employee_id).count()
            return count
        finally:
            self.db_manager.close_session(session)

    # Legacy compatibility methods
    def detect_and_recognize_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Legacy method for backward compatibility"""
        return self.recognize_faces_in_frame(frame)

    def save_image_and_vector(self, image_file, employee_id: int):
        """Legacy method for backward compatibility"""
        result = self.save_employee_image_with_vector(image_file, employee_id)
        
        class MockResult:
            def __init__(self, vector_id):
                self.id = vector_id
        
        if result['success']:
            return MockResult(result['vector_id'])
        else:
            raise Exception(result.get('error', 'Unknown error'))