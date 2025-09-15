# services/face_service.py
import cv2
import numpy as np
from deepface import DeepFace
import os
import uuid
from datetime import datetime
from database.models import VectorFace, Employee
from database.database import DatabaseManager
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceService:
    def __init__(self, upload_folder="static/uploads"):
        self.db_manager = DatabaseManager()
        self.upload_folder = upload_folder
        self.model_name = "Facenet512"  # Có thể thay đổi: VGG-Face, Facenet, OpenFace, etc.
        self.detector_backend = "opencv"  # mtcnn, opencv, ssd, dlib, etc.
        
        # Tạo thư mục upload nếu chưa có
        os.makedirs(upload_folder, exist_ok=True)
    
    def extract_face_vector(self, image_path):
        """
        Trích xuất vector đặc trưng từ ảnh khuôn mặt
        """
        try:
            # Sử dụng DeepFace để trích xuất features
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend
            )
            
            # DeepFace.represent trả về list of dict, lấy embedding đầu tiên
            if isinstance(embedding, list) and len(embedding) > 0:
                return np.array(embedding[0]["embedding"])
            else:
                return np.array(embedding["embedding"])
                
        except Exception as e:
            logger.error(f"Error extracting face vector: {str(e)}")
            raise Exception(f"Cannot extract face from image: {str(e)}")
    
    def detect_faces_in_frame(self, frame):
        """
        Phát hiện khuôn mặt trong frame và trả về bounding boxes
        """
        try:
            # Sử dụng DeepFace để detect faces
            face_objs = DeepFace.extract_faces(
                img=frame,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            # Chuyển đổi về format bounding box
            faces = []
            if face_objs:
                for i, face_obj in enumerate(face_objs):
                    # DeepFace trả về normalized faces, cần tính lại coordinates
                    h, w = frame.shape[:2]
                    # Estimate bounding box (approximation)
                    face_h, face_w = face_obj.shape[:2]
                    # Estimate bounding box coordinates (DeepFace doesn't return exact bbox)
                    x = int(w * 0.1 + i * 100)  # Approximated coordinates
                    y = int(h * 0.1)
                    w_box = int(face_w * 4)  # Scale up
                    h_box = int(face_h * 4)
                    
                    faces.append({
                        'bbox': [x, y, w_box, h_box],
                        'face_img': face_obj
                    })
            
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []
    
    def save_image_and_vector(self, image_data, employee_id):
        """
        Lưu ảnh và vector vào database
        """
        session = self.db_manager.get_session()
        try:
            # Kiểm tra số lượng ảnh hiện có
            current_count = session.query(VectorFace).filter_by(employee_id=employee_id).count()
            if current_count >= 10:
                raise Exception("Employee already has maximum 10 images")
            
            # Tạo tên file unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emp_{employee_id}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
            image_path = os.path.join(self.upload_folder, filename)
            
            # Lưu ảnh
            if isinstance(image_data, np.ndarray):
                cv2.imwrite(image_path, image_data)
            else:
                # Nếu là file upload
                image_data.save(image_path)
            
            # Trích xuất vector
            vector = self.extract_face_vector(image_path)
            
            # Lưu vào database
            vector_face = VectorFace(
                employee_id=employee_id,
                vector_array=vector,
                image_path=image_path
            )
            
            session.add(vector_face)
            session.commit()
            session.refresh(vector_face)
            
            return vector_face
            
        except Exception as e:
            session.rollback()
            # Xóa file nếu có lỗi
            if 'image_path' in locals() and os.path.exists(image_path):
                os.remove(image_path)
            raise e
        finally:
            self.db_manager.close_session(session)
    
    def recognize_face(self, frame):
        """
        Nhận diện khuôn mặt trong frame với database có sẵn
        """
        session = self.db_manager.get_session()
        try:
            # Detect faces trong frame
            faces = self.detect_faces_in_frame(frame)
            if not faces:
                return []
            
            results = []
            
            # Lấy tất cả vectors từ database
            all_vectors = session.query(VectorFace).join(Employee).all()
            
            for face in faces:
                try:
                    # Trích xuất vector từ face hiện tại
                    temp_path = f"temp_{uuid.uuid4().hex}.jpg"
                    face_img = face['face_img']
                    
                    # Convert face_img to proper format for saving
                    if face_img.dtype != np.uint8:
                        face_img = (face_img * 255).astype(np.uint8)
                    
                    cv2.imwrite(temp_path, face_img)
                    current_vector = self.extract_face_vector(temp_path)
                    
                    # So sánh với các vectors trong database
                    best_match = None
                    min_distance = float('inf')
                    threshold = 0.6  # Có thể điều chỉnh
                    
                    for vector_face in all_vectors:
                        stored_vector = np.array(vector_face.vector_array)
                        
                        # Tính cosine similarity hoặc euclidean distance
                        distance = np.linalg.norm(current_vector - stored_vector)
                        
                        if distance < min_distance and distance < threshold:
                            min_distance = distance
                            best_match = vector_face.employee
                    
                    # Cleanup temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    # Thêm kết quả
                    result = {
                        'bbox': face['bbox'],
                        'employee': None,
                        'confidence': 0,
                        'distance': min_distance
                    }
                    
                    if best_match:
                        result['employee'] = {
                            'id': best_match.id,
                            'name': best_match.name,
                            'employee_code': best_match.employee_code,
                            'department': best_match.department
                        }
                        result['confidence'] = max(0.0, float((threshold - min_distance) / threshold))
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing face: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
            return []
        finally:
            self.db_manager.close_session(session)
    
    def delete_face_vector(self, vector_id):
        """
        Xóa vector và ảnh tương ứng
        """
        session = self.db_manager.get_session()
        try:
            vector_face = session.query(VectorFace).filter_by(id=vector_id).first()
            if vector_face:
                # Xóa file ảnh
                if os.path.exists(vector_face.image_path):
                    os.remove(vector_face.image_path)
                
                # Xóa record
                session.delete(vector_face)
                session.commit()
                return True
            return False
        except Exception as e:
            session.rollback()
            raise e
        finally:
            self.db_manager.close_session(session)
    
    def get_employee_vectors(self, employee_id):
        """
        Lấy tất cả vectors của một nhân viên
        """
        session = self.db_manager.get_session()
        try:
            return session.query(VectorFace).filter_by(employee_id=employee_id).all()
        finally:
            self.db_manager.close_session(session)
