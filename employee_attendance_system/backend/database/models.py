# database/models.py
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

Base = declarative_base()

class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    employee_code = Column(String(50), unique=True, nullable=False)
    department = Column(String(100))
    position = Column(String(100))
    email = Column(String(100))
    phone = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    face_vectors = relationship("VectorFace", back_populates="employee", cascade="all, delete-orphan")
    attendance_logs = relationship("AttendanceLog", back_populates="employee")

class VectorFace(Base):
    __tablename__ = 'vector_face'
    
    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey('employees.id'), nullable=False)
    vector_data = Column(Text, nullable=False)  # JSON string
    image_path = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    employee = relationship("Employee", back_populates="face_vectors")
    
    @property
    def vector_array(self):
        return json.loads(self.vector_data)
    
    @vector_array.setter
    def vector_array(self, value):
        self.vector_data = json.dumps(value.tolist() if hasattr(value, 'tolist') else value)

class AttendanceLog(Base):
    __tablename__ = 'attendance_logs'
    
    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, ForeignKey('employees.id'), nullable=False)
    check_in_time = Column(DateTime, default=datetime.utcnow)
    confidence_score = Column(Float)
    image_path = Column(String(255))
    
    # Relationships
    employee = relationship("Employee", back_populates="attendance_logs")
