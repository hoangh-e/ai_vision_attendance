#!/usr/bin/env python3
"""
models.py - DATABASE MODELS
Unified SQLAlchemy models for BHK Tech Attendance System
Optimized for app/ structure with DeepFace integration
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json
import numpy as np

Base = declarative_base()


class Employee(Base):
    """Employee model with enhanced fields"""
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    employee_code = Column(String(50), unique=True, nullable=False)
    department = Column(String(100))
    position = Column(String(100))
    email = Column(String(100))
    phone = Column(String(20))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    face_vectors = relationship("VectorFace", back_populates="employee", cascade="all, delete-orphan")
    attendance_logs = relationship("AttendanceLog", back_populates="employee")
    
    def __repr__(self):
        return f"<Employee(id={self.id}, name='{self.name}', code='{self.employee_code}')>"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'employee_code': self.employee_code,
            'department': self.department,
            'position': self.position,
            'email': self.email,
            'phone': self.phone,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'face_vector_count': len(self.face_vectors) if self.face_vectors else 0
        }


class VectorFace(Base):
    """Face vector model with DeepFace integration"""
    __tablename__ = 'vector_face'
    
    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey('employees.id'), nullable=False)
    vector_data = Column(Text, nullable=False)  # JSON string of face embedding
    image_path = Column(String(255), nullable=False)
    model_name = Column(String(50), default='Facenet512')  # DeepFace model used
    detection_backend = Column(String(50), default='opencv')  # Detection backend used
    confidence_score = Column(Float)  # Detection confidence
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    employee = relationship("Employee", back_populates="face_vectors")
    
    @property
    def vector_array(self):
        """Get vector as numpy array"""
        try:
            return np.array(json.loads(self.vector_data))
        except (json.JSONDecodeError, ValueError):
            return None
    
    @vector_array.setter
    def vector_array(self, value):
        """Set vector from numpy array or list"""
        if value is not None:
            if hasattr(value, 'tolist'):
                self.vector_data = json.dumps(value.tolist())
            else:
                self.vector_data = json.dumps(list(value))
        else:
            self.vector_data = None
    
    def __repr__(self):
        return f"<VectorFace(id={self.id}, employee_id={self.employee_id}, model='{self.model_name}')>"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'image_path': self.image_path,
            'model_name': self.model_name,
            'detection_backend': self.detection_backend,
            'confidence_score': self.confidence_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'vector_size': len(self.vector_array) if self.vector_array is not None else 0
        }


class AttendanceLog(Base):
    """Attendance log model with real-time detection results"""
    __tablename__ = 'attendance_logs'
    
    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey('employees.id'), nullable=False)
    check_in_time = Column(DateTime, default=datetime.utcnow)
    confidence_score = Column(Float)
    image_path = Column(String(255))
    detection_method = Column(String(50), default='realtime')  # realtime, manual, etc.
    client_id = Column(String(100))  # Mobile client ID
    processing_time = Column(Float)  # Time taken for face recognition
    bbox_data = Column(Text)  # JSON string of bounding box coordinates
    
    # Relationships
    employee = relationship("Employee", back_populates="attendance_logs")
    
    @property
    def bbox_coordinates(self):
        """Get bounding box as list [x, y, w, h]"""
        try:
            return json.loads(self.bbox_data) if self.bbox_data else None
        except (json.JSONDecodeError, ValueError):
            return None
    
    @bbox_coordinates.setter
    def bbox_coordinates(self, value):
        """Set bounding box from list [x, y, w, h]"""
        if value is not None:
            self.bbox_data = json.dumps(list(value))
        else:
            self.bbox_data = None
    
    def __repr__(self):
        return f"<AttendanceLog(id={self.id}, employee_id={self.employee_id}, time={self.check_in_time})>"
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'employee_name': self.employee.name if self.employee else None,
            'check_in_time': self.check_in_time.isoformat() if self.check_in_time else None,
            'confidence_score': self.confidence_score,
            'image_path': self.image_path,
            'detection_method': self.detection_method,
            'client_id': self.client_id,
            'processing_time': self.processing_time,
            'bbox_coordinates': self.bbox_coordinates
        }


class SystemSettings(Base):
    """System settings and configuration"""
    __tablename__ = 'system_settings'
    
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text)
    description = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<SystemSettings(key='{self.key}', value='{self.value}')>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'key': self.key,
            'value': self.value,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class StreamSession(Base):
    """Real-time streaming session tracking"""
    __tablename__ = 'stream_sessions'
    
    id = Column(Integer, primary_key=True)
    client_id = Column(String(100), unique=True, nullable=False)
    client_type = Column(String(20))  # mobile, desktop
    user_agent = Column(String(255))
    screen_size = Column(String(50))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    total_frames = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    avg_processing_time = Column(Float)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<StreamSession(client_id='{self.client_id}', type='{self.client_type}')>"
    
    def to_dict(self):
        return {
            'id': self.id,
            'client_id': self.client_id,
            'client_type': self.client_type,
            'user_agent': self.user_agent,
            'screen_size': self.screen_size,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'avg_processing_time': self.avg_processing_time,
            'is_active': self.is_active,
            'duration_seconds': (
                (self.end_time - self.start_time).total_seconds() 
                if self.end_time and self.start_time 
                else (datetime.utcnow() - self.start_time).total_seconds() 
                if self.start_time else 0
            )
        }


# Database Manager class
class DatabaseManager:
    """Enhanced database manager with connection pooling and error handling"""
    
    def __init__(self, database_url=None):
        from config import Config
        
        self.database_url = database_url or Config.get_database_url()
        self.engine = create_engine(
            self.database_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_pre_ping=True  # Validate connections before use
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        self.create_tables()
    
    def create_tables(self):
        """Create all tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
            print("✅ Database tables created/verified successfully")
        except Exception as e:
            print(f"❌ Error creating database tables: {e}")
            raise
    
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close database session"""
        if session:
            session.close()
    
    def get_db_stats(self):
        """Get database statistics"""
        session = self.get_session()
        try:
            stats = {
                'total_employees': session.query(Employee).count(),
                'active_employees': session.query(Employee).filter(Employee.is_active == True).count(),
                'total_face_vectors': session.query(VectorFace).count(),
                'total_attendance_logs': session.query(AttendanceLog).count(),
                'active_stream_sessions': session.query(StreamSession).filter(StreamSession.is_active == True).count(),
                'database_url': self.database_url,
                'engine_info': str(self.engine.url)
            }
            return stats
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return {'error': str(e)}
        finally:
            self.close_session(session)
    
    def test_connection(self):
        """Test database connection"""
        try:
            session = self.get_session()
            session.execute("SELECT 1")
            session.close()
            return True
        except Exception as e:
            print(f"❌ Database connection test failed: {e}")
            return False
    
    def reset_database(self):
        """Reset database (DROP ALL TABLES and recreate)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            Base.metadata.create_all(bind=self.engine)
            print("✅ Database reset successfully")
            return True
        except Exception as e:
            print(f"❌ Error resetting database: {e}")
            return False


# Convenience functions for common database operations
def get_db_session():
    """Get database session (for use in services)"""
    from config import Config
    db_manager = DatabaseManager()
    return db_manager.get_session()


def init_default_settings():
    """Initialize default system settings"""
    session = get_db_session()
    try:
        default_settings = [
            {
                'key': 'face_recognition_threshold',
                'value': '0.6',
                'description': 'Face recognition confidence threshold'
            },
            {
                'key': 'max_images_per_employee',
                'value': '10',
                'description': 'Maximum face images per employee'
            },
            {
                'key': 'stream_max_clients',
                'value': '20',
                'description': 'Maximum concurrent streaming clients'
            },
            {
                'key': 'detection_active',
                'value': 'true',
                'description': 'Global face detection enabled status'
            }
        ]
        
        for setting_data in default_settings:
            existing = session.query(SystemSettings).filter(
                SystemSettings.key == setting_data['key']
            ).first()
            
            if not existing:
                setting = SystemSettings(**setting_data)
                session.add(setting)
        
        session.commit()
        print("✅ Default system settings initialized")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error initializing default settings: {e}")
    finally:
        session.close()


# Export all models and manager
__all__ = [
    'Base',
    'Employee', 
    'VectorFace', 
    'AttendanceLog', 
    'SystemSettings', 
    'StreamSession',
    'DatabaseManager',
    'get_db_session',
    'init_default_settings'
]