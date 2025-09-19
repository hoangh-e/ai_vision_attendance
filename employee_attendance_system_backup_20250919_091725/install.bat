@echo off
echo 🚀 Bat dau cai dat he thong diem danh...

REM Kiem tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python chua duoc cai dat
    pause
    exit /b 1
)

echo ✅ Python da co san

REM Tao virtual environment
echo 📦 Tao virtual environment...
python -m venv venv

REM Kich hoat virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Cap nhat pip...
python -m pip install --upgrade pip

REM Cai dat requirements
echo 📥 Cai dat dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Loi cai dat voi version co dinh
    echo 🔧 Thu cai dat voi requirements minimal...
    pip install -r requirements_minimal.txt
    if errorlevel 1 (
        echo ❌ Van loi, cai dat tung package rieng le...
        pip install Flask
        pip install Flask-SocketIO
        pip install SQLAlchemy
        pip install opencv-python
        pip install deepface
        pip install tensorflow
        pip install numpy
        pip install Pillow
        pip install python-socketio
        pip install eventlet
        pip install protobuf
        pip install requests
    )
)

REM Chay setup
echo ⚙️ Thiet lap project...
python setup.py
if errorlevel 1 (
    echo ❌ Loi setup project
    echo 💡 Ban co the chay thu cong: python setup.py
)

echo 🎉 Cai dat hoan tat!
echo.
echo 🚀 De chay ung dung:
echo 1. Kich hoat virtual environment: venv\Scripts\activate.bat
echo 2. Chuyen vao thu muc backend: cd backend
echo 3. Chay server: python app.py
echo 4. Mo trinh duyet: http://localhost:5000 (laptop)
echo 5. Mobile app: http://[YOUR_IP]:5000/mobile (dien thoai)
echo.
echo 💡 Lay IP laptop: ipconfig
pause