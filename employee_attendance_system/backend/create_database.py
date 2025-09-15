# Database Schema (SQLite)
import sqlite3
from datetime import datetime

def create_database():
    conn = sqlite3.connect('attendance_system.db')
    cursor = conn.cursor()
    
    # Bảng Nhân viên
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        employee_code TEXT UNIQUE NOT NULL,
        department TEXT,
        position TEXT,
        email TEXT,
        phone TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Bảng vector_face
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vector_face (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER NOT NULL,
        vector_data TEXT NOT NULL,  -- JSON string của vector
        image_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (employee_id) REFERENCES employees (id) ON DELETE CASCADE
    )
    ''')
    
    # Bảng lịch sử điểm danh (optional)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER NOT NULL,
        check_in_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        confidence_score REAL,
        image_path TEXT,
        FOREIGN KEY (employee_id) REFERENCES employees (id)
    )
    ''')
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    print("Database created successfully!")