#!/usr/bin/env python3
"""
config.py - GLOBAL CONFIGURATION
Centralized configuration management for the BHK Tech Attendance System
"""

import os
import logging
from datetime import timedelta
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent.parent.absolute()


class Config:
    """Base configuration class"""
    
    # =====================================================
    # APPLICATION SETTINGS
    # =====================================================
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'BHK-Tech-Attendance-System-2024'
    DEBUG = False
    TESTING = False
    
    # =====================================================
    # SERVER SETTINGS
    # =====================================================
    HOST = os.environ.get('HOST') or '0.0.0.0'
    PORT = int(os.environ.get('PORT') or 5000)
    
    # =====================================================
    # FILE UPLOAD SETTINGS
    # =====================================================
    UPLOAD_FOLDER = BASE_DIR / 'frontend' / 'static' / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    MAX_IMAGES_PER_EMPLOYEE = 10
    
    # =====================================================
    # DATABASE SETTINGS
    # =====================================================
    DATABASE_URL = os.environ.get('DATABASE_URL') or f'sqlite:///{BASE_DIR / "attendance_system.db"}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_RECORD_QUERIES = True
    
    # =====================================================
    # FACE RECOGNITION SETTINGS
    # =====================================================
    FACE_RECOGNITION_MODEL = os.environ.get('FACE_MODEL') or 'Facenet512'
    FACE_DETECTION_BACKEND = os.environ.get('FACE_BACKEND') or 'opencv'
    FACE_RECOGNITION_THRESHOLD = float(os.environ.get('FACE_THRESHOLD') or 0.6)
    FACE_MAX_SIZE = (300, 300)
    FACE_DETECTION_SCALE_FACTOR = 1.1
    FACE_MIN_NEIGHBORS = 5
    
    # =====================================================
    # VIDEO STREAMING SETTINGS
    # =====================================================
    STREAM_MAX_CLIENTS = int(os.environ.get('STREAM_MAX_CLIENTS') or 10)
    STREAM_FRAME_TIMEOUT = int(os.environ.get('STREAM_TIMEOUT') or 30)  # seconds
    STREAM_CLEANUP_INTERVAL = int(os.environ.get('STREAM_CLEANUP') or 60)  # seconds
    STREAM_MAX_FPS = int(os.environ.get('STREAM_MAX_FPS') or 15)
    
    # =====================================================
    # SOCKETIO SETTINGS
    # =====================================================
    SOCKETIO_ASYNC_MODE = 'threading'
    SOCKETIO_CORS_ALLOWED_ORIGINS = '*'
    SOCKETIO_PING_TIMEOUT = 60
    SOCKETIO_PING_INTERVAL = 25
    
    # =====================================================
    # LOGGING SETTINGS
    # =====================================================
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_DIR = BASE_DIR / 'logs'
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # =====================================================
    # SECURITY SETTINGS
    # =====================================================
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # =====================================================
    # PERFORMANCE SETTINGS
    # =====================================================
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    
    # =====================================================
    # API SETTINGS
    # =====================================================
    API_RATE_LIMIT = os.environ.get('API_RATE_LIMIT') or '100 per hour'
    API_RESULTS_PER_PAGE = int(os.environ.get('API_PAGE_SIZE') or 50)
    
    @staticmethod
    def init_app(app):
        """Initialize application configuration"""
        # Create necessary directories
        Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        Config.setup_logging()
    
    @staticmethod
    def setup_logging():
        """Setup application logging"""
        # Create logs directory if it doesn't exist
        Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL.upper()),
            format=Config.LOG_FORMAT,
            handlers=[
                logging.FileHandler(Config.LOG_DIR / 'app.log'),
                logging.StreamHandler()
            ]
        )
        
        # Setup specific loggers
        loggers = {
            'face_recognition': Config.LOG_DIR / 'face_recognition.log',
            'stream_service': Config.LOG_DIR / 'stream.log',
            'employee_service': Config.LOG_DIR / 'employee.log',
            'werkzeug': Config.LOG_DIR / 'access.log'
        }
        
        for logger_name, log_file in loggers.items():
            logger = logging.getLogger(logger_name)
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
            logger.addHandler(handler)
    
    @classmethod
    def get_database_url(cls):
        """Get properly formatted database URL"""
        return str(cls.DATABASE_URL).replace('sqlite:///', f'sqlite:///{BASE_DIR}/')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
    # More verbose logging in development
    LOG_LEVEL = 'DEBUG'
    
    # Relaxed settings for development
    FACE_RECOGNITION_THRESHOLD = 0.5
    STREAM_MAX_CLIENTS = 5
    
    # Development database
    DATABASE_URL = f'sqlite:///{BASE_DIR / "attendance_system_dev.db"}'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    
    # Use in-memory database for testing
    DATABASE_URL = 'sqlite:///:memory:'
    
    # Disable logging during tests
    LOG_LEVEL = 'CRITICAL'
    
    # Test-specific settings
    WTF_CSRF_ENABLED = False
    FACE_RECOGNITION_THRESHOLD = 0.8


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
    # Stricter settings for production
    SESSION_COOKIE_SECURE = True  # Requires HTTPS
    FACE_RECOGNITION_THRESHOLD = 0.6
    LOG_LEVEL = 'WARNING'
    
    # Production database (should be set via environment variable)
    DATABASE_URL = os.environ.get('DATABASE_URL') or f'sqlite:///{BASE_DIR / "attendance_system_prod.db"}'
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Production-specific initialization
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Setup rotating log files for production
        if not app.debug:
            file_handler = RotatingFileHandler(
                Config.LOG_DIR / 'app.log',
                maxBytes=Config.LOG_MAX_BYTES,
                backupCount=Config.LOG_BACKUP_COUNT
            )
            file_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


def get_config(config_name=None):
    """Get configuration class based on environment"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default').lower()
    
    return config.get(config_name, config['default'])


# Convenience functions for common configurations
def is_development():
    """Check if running in development mode"""
    return os.environ.get('FLASK_ENV', 'development').lower() == 'development'


def is_production():
    """Check if running in production mode"""
    return os.environ.get('FLASK_ENV', 'development').lower() == 'production'


def get_upload_path():
    """Get the upload directory path"""
    config_class = get_config()
    return str(config_class.UPLOAD_FOLDER)


def get_log_path():
    """Get the log directory path"""
    config_class = get_config()
    return str(config_class.LOG_DIR)
