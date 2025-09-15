#!/bin/bash
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
