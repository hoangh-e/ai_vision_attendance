@echo off
echo Starting BHK Tech Attendance System...
echo.

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
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
