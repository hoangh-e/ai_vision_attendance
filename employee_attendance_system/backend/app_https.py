# app_https.py - HTTPS version of the Flask app
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from services.face_service import FaceService
from services.employee_service import EmployeeService
import json
import os
from werkzeug.utils import secure_filename
import threading
import time
import ssl

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'BHK-Tech-Attendance-System-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Initialize SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize services
face_service = FaceService()
employee_service = EmployeeService()

# Global variables for real-time processing
detection_active = False
camera_active = False
current_frame = None

class WebRTCService:
    def __init__(self):
        self.clients = {}
        self.processing_active = False
    
    def register_client(self, client_id, socket_id):
        self.clients[client_id] = socket_id
    
    def unregister_client(self, client_id):
        if client_id in self.clients:
            del self.clients[client_id]
    
    def broadcast_to_clients(self, data):
        for client_id, socket_id in self.clients.items():
            socketio.emit('detection_result', data, room=socket_id)

webrtc_service = WebRTCService()

# Import all routes from main app
import sys
import importlib.util

# Load routes from app.py
spec = importlib.util.spec_from_file_location("main_app", "app.py")
main_app = importlib.util.module_from_spec(spec)

# Copy routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    employees = employee_service.get_all_employees()
    return render_template('dashboard.html', employees=employees)

@app.route('/mobile')
def mobile_app():
    return app.send_static_file('mobile_app/index.html')

@app.route('/mobile/simple')
def mobile_app_simple():
    return app.send_static_file('mobile_app/simple.html')

@app.route('/api/employees')
def get_employees():
    """Get list of employees for testing"""
    return jsonify({
        'employees': [
            {'id': 1, 'name': 'Nguyễn Văn A', 'employee_id': 'EMP001', 'department': 'IT'},
            {'id': 2, 'name': 'Trần Thị B', 'employee_id': 'EMP002', 'department': 'HR'},
            {'id': 3, 'name': 'Lê Văn C', 'employee_id': 'EMP003', 'department': 'Finance'}
        ]
    })

@app.route('/api/employees', methods=['POST'])
def add_employee():
    """Add new employee"""
    data = request.get_json()
    return jsonify({
        'success': True,
        'message': 'Employee added successfully',
        'employee': {
            'id': 4,
            'name': data.get('name', 'Unknown'),
            'employee_id': data.get('employee_id', 'NEW001'),
            'department': data.get('department', 'General')
        }
    })

@app.route('/api/process_face', methods=['POST'])
def process_face_api():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Không có dữ liệu hình ảnh'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'message': 'Không thể decode hình ảnh'}), 400
        
        # Simple mock response for testing
        return jsonify({
            'success': True,
            'employee': {
                'name': 'Test User',
                'employee_id': 'TEST001',
                'department': 'IT'
            },
            'confidence': 0.95,
            'timestamp': data.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S')),
            'message': 'Demo mode - Face recognition working!'
        })
            
    except Exception as e:
        print(f"Error in process_face_api: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Lỗi server: {str(e)}'
        }), 500

# Socket events
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")

@socketio.on('process_frame')
def handle_process_frame(data):
    try:
        # Simple echo for testing
        emit('detection_result', {
            'success': True,
            'employee': {
                'name': 'Demo User',
                'employee_id': 'DEMO001',
                'department': 'Testing'
            },
            'confidence': 0.9,
            'timestamp': data.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S'))
        })
    except Exception as e:
        emit('detection_result', {
            'success': False,
            'message': f'Error: {str(e)}'
        })

if __name__ == '__main__':
    # Tạo thư mục uploads nếu chưa có
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("Starting HTTPS Flask server...")
    print("Access laptop interface at: https://localhost:5000")
    print("Access mobile interface at: https://localhost:5000/mobile")
    print("Simple mobile interface at: https://localhost:5000/mobile/simple")
    print("")
    print("🔒 HTTPS enabled for mobile camera access!")
    print("⚠️  You may see security warnings - click 'Advanced' -> 'Proceed to localhost'")
    print("")
    
    # Try different HTTPS approaches
    try:
        print("Creating adhoc SSL certificate...")
        # Try with Flask's HTTPS first (simpler)
        app.run(host='0.0.0.0', 
                port=5000, 
                debug=True,
                ssl_context='adhoc',
                threaded=True)
        
    except Exception as e:
        print(f"Error starting HTTPS server: {e}")
        print("Trying SocketIO without SSL...")
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)