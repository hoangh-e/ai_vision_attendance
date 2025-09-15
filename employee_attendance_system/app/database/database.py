#!/usr/bin/env python3
"""
database.py - DATABASE CONNECTION MANAGER
Enhanced database management for BHK Tech Attendance System
Unified database operations with connection pooling
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
import logging
import os
from datetime import datetime

# Import models
from .models import Base, Employee, VectorFace, AttendanceLog, SystemSettings, StreamSession

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Enhanced database manager with connection pooling and error handling"""
    
    def __init__(self, database_url=None):
        from ..config import Config
        
        self.database_url = database_url or Config.get_database_url()
        self._setup_engine()
        self._setup_session_factory()
        self.create_tables()
    
    def _setup_engine(self):
        """Setup SQLAlchemy engine with optimized settings"""
        try:
            # SQLite specific optimizations
            if 'sqlite' in self.database_url.lower():
                self.engine = create_engine(
                    self.database_url,
                    echo=False,
                    poolclass=StaticPool,
                    connect_args={
                        'check_same_thread': False,
                        'timeout': 20,
                        'isolation_level': None  # Autocommit mode
                    },
                    pool_pre_ping=True
                )
            else:
                # For other databases (PostgreSQL, MySQL, etc.)
                self.engine = create_engine(
                    self.database_url,
                    echo=False,
                    pool_size=10,
                    max_overflow=20,
                    pool_recycle=3600,
                    pool_pre_ping=True
                )
            
            logger.info(f"✅ Database engine created: {self.database_url}")
            
        except Exception as e:
            logger.error(f"❌ Error creating database engine: {e}")
            raise
    
    def _setup_session_factory(self):
        """Setup session factory"""
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """Create all tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database tables created/verified successfully")
            
            # Initialize default settings after creating tables
            self._init_default_settings()
            
        except Exception as e:
            logger.error(f"❌ Error creating database tables: {e}")
            raise
    
    def _init_default_settings(self):
        """Initialize default system settings"""
        try:
            session = self.get_session()
            
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
                },
                {
                    'key': 'deepface_model',
                    'value': 'Facenet512',
                    'description': 'DeepFace model for face recognition'
                },
                {
                    'key': 'detection_backend',
                    'value': 'opencv',
                    'description': 'Face detection backend'
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
            logger.info("✅ Default system settings initialized")
            
        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"❌ Error initializing default settings: {e}")
        finally:
            if session:
                session.close()
    
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def close_session(self, session):
        """Close database session"""
        if session:
            session.close()
    
    def test_connection(self):
        """Test database connection"""
        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            logger.info("✅ Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    def get_db_stats(self):
        """Get comprehensive database statistics"""
        try:
            with self.session_scope() as session:
                stats = {
                    'total_employees': session.query(Employee).count(),
                    'active_employees': session.query(Employee).filter(
                        Employee.is_active == True
                    ).count(),
                    'total_face_vectors': session.query(VectorFace).count(),
                    'total_attendance_logs': session.query(AttendanceLog).count(),
                    'active_stream_sessions': session.query(StreamSession).filter(
                        StreamSession.is_active == True
                    ).count(),
                    'total_system_settings': session.query(SystemSettings).count(),
                    'database_url': self.database_url,
                    'engine_info': str(self.engine.url),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Get employee statistics by department
                departments = session.query(Employee.department).distinct().all()
                dept_stats = {}
                for dept in departments:
                    if dept[0]:
                        dept_count = session.query(Employee).filter(
                            Employee.department == dept[0]
                        ).count()
                        dept_stats[dept[0]] = dept_count
                stats['employees_by_department'] = dept_stats
                
                # Get recent activity
                recent_logs = session.query(AttendanceLog).order_by(
                    AttendanceLog.check_in_time.desc()
                ).limit(5).all()
                
                stats['recent_activity'] = [
                    {
                        'employee_name': log.employee.name if log.employee else 'Unknown',
                        'check_in_time': log.check_in_time.isoformat() if log.check_in_time else None,
                        'confidence_score': log.confidence_score
                    }
                    for log in recent_logs
                ]
                
                return stats
                
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    def reset_database(self):
        """Reset database (DROP ALL TABLES and recreate)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            Base.metadata.create_all(bind=self.engine)
            self._init_default_settings()
            logger.info("✅ Database reset successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error resetting database: {e}")
            return False
    
    def backup_database(self, backup_path=None):
        """Create database backup (SQLite only)"""
        if 'sqlite' not in self.database_url.lower():
            logger.warning("Database backup only supported for SQLite")
            return False
        
        try:
            import shutil
            from datetime import datetime
            
            # Extract database file path from URL
            db_file = self.database_url.replace('sqlite:///', '')
            
            if not backup_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"{db_file}.backup_{timestamp}"
            
            shutil.copy2(db_file, backup_path)
            logger.info(f"✅ Database backed up to: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"❌ Error backing up database: {e}")
            return False
    
    def get_table_info(self):
        """Get information about all tables"""
        try:
            with self.session_scope() as session:
                tables_info = {}
                
                for table_name, table in Base.metadata.tables.items():
                    row_count = session.execute(
                        text(f"SELECT COUNT(*) FROM {table_name}")
                    ).scalar()
                    
                    tables_info[table_name] = {
                        'row_count': row_count,
                        'columns': [col.name for col in table.columns],
                        'primary_keys': [col.name for col in table.primary_key.columns]
                    }
                
                return tables_info
                
        except Exception as e:
            logger.error(f"❌ Error getting table info: {e}")
            return {'error': str(e)}
    
    def optimize_database(self):
        """Optimize database performance (SQLite VACUUM)"""
        if 'sqlite' not in self.database_url.lower():
            logger.warning("Database optimization only supported for SQLite")
            return False
        
        try:
            with self.engine.connect() as connection:
                connection.execute(text("VACUUM"))
                connection.execute(text("ANALYZE"))
            
            logger.info("✅ Database optimized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error optimizing database: {e}")
            return False


# Global database manager instance
_db_manager = None

def get_database_manager():
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_db_session():
    """Get database session (for use in services)"""
    return get_database_manager().get_session()


@contextmanager
def db_session_scope():
    """Provide a transactional scope around a series of operations"""
    db_manager = get_database_manager()
    session = db_manager.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Database health check
def check_database_health():
    """Check database health and return status"""
    try:
        db_manager = get_database_manager()
        
        # Test connection
        if not db_manager.test_connection():
            return {'status': 'unhealthy', 'error': 'Connection failed'}
        
        # Get stats
        stats = db_manager.get_db_stats()
        if 'error' in stats:
            return {'status': 'unhealthy', 'error': stats['error']}
        
        return {
            'status': 'healthy',
            'stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Database health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }


# Export all database utilities
__all__ = [
    'DatabaseManager',
    'get_database_manager',
    'get_db_session',
    'db_session_scope',
    'check_database_health'
]