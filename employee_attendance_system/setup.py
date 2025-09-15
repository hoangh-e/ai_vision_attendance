#!/usr/bin/env python3
"""
setup.py - Khởi tạo hệ thống điểm danh BHK Tech
Tối ưu hóa cho cấu trúc project mới
"""

import os
import sys
import sqlite3
import subprocess

def create_directory_structure():
    """Tạo cấu trúc thư mục cần thiết"""
    directories = [
        'backend/database',
        'backend/services', 
        'backend/api',
        'frontend/templates',
        'frontend/static/css',
        'frontend/static/js',
        'frontend/static/uploads',
        'logs'
    ]
    
    print("🏗️  Tạo cấu trúc thư mục...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")

def create_init_files():
    """Tạo các file __init__.py"""
    init_files = [
        'backend/__init__.py',
        'backend/database/__init__.py',
        'backend/services/__init__.py',
        'backend/api/__init__.py'
    ]
    
    print("\n📄 Tạo file __init__.py...")
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# This file makes Python treat the directory as a package\n')
            print(f"✅ Created: {init_file}")

def setup_database():
    """Khởi tạo database"""
    print("\n💾 Khởi tạo database...")
    
    try:
        # Add backend to Python path for imports
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
        
        from backend.database.database import DatabaseManager
        db_manager = DatabaseManager()
        print("✅ Database initialized successfully")
        
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        print("💡 Sẽ thử tạo database cơ bản...")
        
        # Fallback: Create basic database
        return create_basic_database()

def create_basic_database():
    """Tạo database cơ bản nếu SQLAlchemy không hoạt động"""
    try:
        conn = sqlite3.connect('attendance_system.db')
        cursor = conn.cursor()
        
        # Tạo bảng employees
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
        
        # Tạo bảng vector_face
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS vector_face (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            vector_data TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees (id) ON DELETE CASCADE
        )
        ''')
        
        # Tạo bảng attendance_logs
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
        
        print("✅ Basic database created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Basic database creation failed: {e}")
        return False

def create_demo_data():
    """Tạo dữ liệu demo cho testing"""
    print("\n👥 Tạo dữ liệu demo...")
    
    try:
        conn = sqlite3.connect('attendance_system.db')
        cursor = conn.cursor()
        
        # Kiểm tra xem đã có dữ liệu chưa
        cursor.execute("SELECT COUNT(*) FROM employees")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Thêm nhân viên demo
            demo_employees = [
                ('Nguyễn Văn A', 'EMP001', 'IT', 'Software Engineer', 'nguyenvana@bhktech.com', '0901234567'),
                ('Trần Thị B', 'EMP002', 'HR', 'HR Manager', 'tranthib@bhktech.com', '0901234568'),
                ('Lê Văn C', 'EMP003', 'Finance', 'Accountant', 'levanc@bhktech.com', '0901234569')
            ]
            
            cursor.executemany('''
                INSERT INTO employees (name, employee_code, department, position, email, phone)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', demo_employees)
            
            conn.commit()
            print(f"✅ Added {len(demo_employees)} demo employees")
        else:
            print(f"ℹ️  Database already has {count} employees")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Demo data creation failed: {e}")
        return False

def check_dependencies():
    """Kiểm tra dependencies quan trọng"""
    print("\n🔍 Kiểm tra dependencies...")
    
    critical_packages = [
        'flask', 'flask_socketio', 'cv2', 'numpy', 'sqlalchemy'
    ]
    
    missing_packages = []
    
    for package in critical_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'flask_socketio':
                import flask_socketio
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    print("✅ All critical dependencies available")
    return True

def create_startup_script():
    """Tạo script khởi động nhanh"""
    print("\n🚀 Tạo startup script...")
    
    # Windows batch file (without emojis to avoid encoding issues)
    with open('start.bat', 'w', encoding='utf-8') as f:
        f.write('''@echo off
echo Starting BHK Tech Attendance System...
echo.

REM Activate virtual environment if exists
if exist venv\\Scripts\\activate.bat (
    call venv\\Scripts\\activate.bat
    echo Virtual environment activated
) else (
    echo No virtual environment found
)

echo.
echo Desktop Interface: http://localhost:5000
echo Mobile Interface:  http://[YOUR_IP]:5000/mobile
echo.
echo Get your IP: ipconfig
echo.

python main_app.py
pause
''')
    
    # Linux/Mac shell script
    with open('start.sh', 'w', encoding='utf-8') as f:
        f.write('''#!/bin/bash
echo "Starting BHK Tech Attendance System..."
echo

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
else
    echo "No virtual environment found"
fi

echo
echo "Desktop Interface: http://localhost:5000"
echo "Mobile Interface:  http://[YOUR_IP]:5000/mobile"
echo
echo "Get your IP: ifconfig"
echo

python3 main_app.py
''')
    
    # Make shell script executable
    try:
        os.chmod('start.sh', 0o755)
    except:
        pass
    
    print("✅ Created start.bat and start.sh")

def display_completion_message():
    """Hiển thị thông báo hoàn thành"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print()
    print("📋 QUICK START GUIDE:")
    print("1. 🚀 Run server:")
    print("   Windows: start.bat")
    print("   Linux/Mac: ./start.sh")
    print("   Manual: python main_app.py")
    print()
    print("2. 🖥️  Open desktop interface:")
    print("   http://localhost:5000")
    print()
    print("3. 📱 Mobile access:")
    print("   http://[YOUR_IP]:5000/mobile")
    print("   (Replace [YOUR_IP] with your computer's IP)")
    print()
    print("4. 💡 Get your IP address:")
    print("   Windows: ipconfig")
    print("   Linux/Mac: ifconfig")
    print()
    print("📂 PROJECT STRUCTURE:")
    print("├── main_app.py           # 🎯 Main application")
    print("├── backend/              # 🔧 Core services")
    print("├── frontend/             # 🎨 Web interfaces")
    print("└── start.bat/.sh         # 🚀 Quick start scripts")
    print()
    print("🔗 URLS:")
    print("├── Desktop: http://localhost:5000")
    print("├── Mobile:  http://[YOUR_IP]:5000/mobile")
    print("└── API:     http://localhost:5000/api/employees")
    print()
    print("✅ Ready to use!")
    print("="*60)

def main():
    """Main setup function"""
    print("🏗️  BHK TECH ATTENDANCE SYSTEM SETUP")
    print("="*50)
    
    # Step 1: Create directory structure
    create_directory_structure()
    
    # Step 2: Create __init__.py files
    create_init_files()
    
    # Step 3: Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Setup incomplete due to missing dependencies")
        print("💡 Please install requirements: pip install -r requirements.txt")
        return False
    
    # Step 4: Setup database
    db_ok = setup_database()
    if not db_ok:
        print("\n❌ Database setup failed")
        return False
    
    # Step 5: Create demo data
    create_demo_data()
    
    # Step 6: Create startup scripts
    create_startup_script()
    
    # Step 7: Display completion message
    display_completion_message()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)