# setup.py
import os
import sys
import sqlite3

# Add backend to Python path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from database.database import DatabaseManager

def setup_project():
    """Thiết lập project lần đầu"""
    print("🚀 Đang thiết lập project...")
    
    # Tạo các thư mục cần thiết
    directories = [
        'static/uploads',
        'static/css',
        'static/js',
        'templates',
        'database',
        'services',
        'api'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Tạo thư mục: {directory}")
    
    # Khởi tạo database
    try:
        db_manager = DatabaseManager()
        print("✅ Database đã được khởi tạo")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo database: {e}")
        return False
    
    # Tạo file __init__.py
    init_files = [
        'database/__init__.py',
        'services/__init__.py',
        'api/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# This file makes Python treat the directory as a package\n')
            print(f"✅ Tạo file: {init_file}")
    
    print("🎉 Setup hoàn tất!")
    return True

if __name__ == "__main__":
    setup_project()