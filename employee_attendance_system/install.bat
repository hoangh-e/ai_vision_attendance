@echo off
echo 🚀 Bắt đầu cài đặt hệ thống điểm danh...

REM Kiểm tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python chưa được cài đặt
    pause
    exit /b 1
)

echo ✅ Python đã có sẵn

REM Tạo virtual environment
echo 📦 Tạo virtual environment...
python -m venv venv

REM Kích hoạt virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Cập nhật pip...
python -m pip install --upgrade pip

REM Cài đặt requirements
echo 📥 Cài đặt dependencies...
pip install -r requirements.txt

REM Chạy setup
echo ⚙️ Thiết lập project...
python setup.py

echo 🎉 Cài đặt hoàn tất!
echo.
echo 🚀 Để chạy ứng dụng:
echo 1. Kích hoạt virtual environment: venv\Scripts\activate.bat
echo 2. Chạy server: python app.py
echo 3. Mở trình duyệt: http://localhost:5000
echo 4. Mobile app: http://localhost:5000/mobile
pause