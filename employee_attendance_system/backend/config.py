# config.py
import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'BHK-Tech-Attendance-System-2024'
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///attendance_system.db'
    
    # Face recognition settings
    FACE_RECOGNITION_MODEL = 'Facenet512'
    FACE_DETECTION_BACKEND = 'opencv'
    FACE_RECOGNITION_THRESHOLD = 0.6
    MAX_IMAGES_PER_EMPLOYEE = 10
    
    # WebRTC settings
    WEBRTC_STUN_SERVER = 'stun:stun.l.google.com:19302'
    FRAME_RATE = 10
    
    # Logging
    LOG_LEVEL = 'INFO'

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
