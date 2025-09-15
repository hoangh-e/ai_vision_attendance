# app.py
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

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/management')
def management():
    employees = employee_service.get_all_employees()
    return render_template('management.html', employees=employees)

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/capture')
def capture():
    employees = employee_service.get_all_employees()
    return render_template('capture.html', employees=employees)

# API Routes for Employee Management
@app.route('/api/employees', methods=['GET'])
def get_employees():
    try:
        employees = employee_service.get_all_employees()
        result = []
        for emp in employees:
            result.append({
                'id': emp.id,
                'name': emp.name,
                'employee_code': emp.employee_code,
                'department': emp.department,
                'position': emp.position,
                'email': emp.email,
                'phone': emp.phone,
                'image_count': len(emp.face_vectors)
            })
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/employees', methods=['POST'])
def create_employee():
    try:
        data = request.json
        employee = employee_service.create_employee(
            name=data['name'],
            employee_code=data['employee_code'],
            department=data.get('department'),
            position=data.get('position'),
            email=data.get('email'),
            phone=data.get('phone')
        )
        return jsonify({'success': True, 'data': {'id': employee.id}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/employees/<int:employee_id>', methods=['PUT'])
def update_employee(employee_id):
    try:
        data = request.json
        employee = employee_service.update_employee(employee_id, **data)
        if employee:
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Employee not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/employees/<int:employee_id>', methods=['DELETE'])
def delete_employee(employee_id):
    try:
        success = employee_service.delete_employee(employee_id)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/employees/<int:employee_id>/upload', methods=['POST'])
def upload_employee_image(employee_id):
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Kiểm tra số lượng ảnh hiện có
        current_count = employee_service.get_employee_image_count(employee_id)
        if current_count >= 10:
            return jsonify({'success': False, 'error': 'Maximum 10 images per employee'})
        
        vector_face = face_service.save_image_and_vector(file, employee_id)
        return jsonify({'success': True, 'data': {'id': vector_face.id}})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/vectors/<int:vector_id>', methods=['DELETE'])
def delete_vector(vector_id):
    try:
        success = face_service.delete_face_vector(vector_id)
        return jsonify({'success': success})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/employees/<int:employee_id>/vectors', methods=['GET'])
def get_employee_vectors(employee_id):
    try:
        vectors = face_service.get_employee_vectors(employee_id)
        result = []
        for vector in vectors:
            result.append({
                'id': vector.id,
                'image_path': vector.image_path,
                'created_at': vector.created_at.isoformat()
            })
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# WebRTC và Real-time Processing
@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

@socketio.on('register_client')
def handle_register_client(data):
    client_id = data.get('client_id', request.sid)
    webrtc_service.register_client(client_id, request.sid)
    emit('registered', {'client_id': client_id})

@socketio.on('camera_frame')
def handle_camera_frame(data):
    global current_frame, detection_active
    try:
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        current_frame = frame
        
        # Emit frame back to detection page
        emit('frame_received', {'status': 'received'}, broadcast=True)
        
        # Process if detection is active
        if detection_active:
            results = face_service.recognize_face(frame)
            
            # Send results back to mobile client
            response_data = {
                'detections': results,
                'timestamp': time.time()
            }
            emit('detection_result', response_data)
            
            # Also broadcast to laptop detection page
            emit('detection_update', response_data, broadcast=True)
    
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        emit('error', {'message': str(e)})

@socketio.on('toggle_detection')
def handle_toggle_detection(data):
    global detection_active
    detection_active = data.get('active', False)
    emit('detection_status', {'active': detection_active}, broadcast=True)

@socketio.on('toggle_camera')
def handle_toggle_camera(data):
    global camera_active
    camera_active = data.get('active', False)
    emit('camera_status', {'active': camera_active}, broadcast=True)

@socketio.on('capture_from_laptop')
def handle_laptop_capture(data):
    try:
        employee_id = data['employee_id']
        image_data = data['image']
        
        # Decode and save image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        vector_face = face_service.save_image_and_vector(frame, employee_id)
        
        emit('capture_result', {
            'success': True, 
            'vector_id': vector_face.id,
            'image_count': employee_service.get_employee_image_count(employee_id)
        })
    
    except Exception as e:
        emit('capture_result', {'success': False, 'error': str(e)})

# Static file serving
@app.route('/mobile')
def mobile_app():
    return app.send_static_file('mobile_app/index.html')

@app.route('/mobile/simple')
def mobile_app_simple():
    return app.send_static_file('mobile_app/simple.html')

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
        
        # Process with face recognition
        result = face_service.recognize_face(frame)
        
        if result['success']:
            # Record attendance
            employee_id = result['employee']['employee_id']
            attendance_result = employee_service.record_attendance(employee_id)
            
            return jsonify({
                'success': True,
                'employee': result['employee'],
                'confidence': result['confidence'],
                'timestamp': data.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S')),
                'attendance': attendance_result
            })
        else:
            return jsonify({
                'success': False,
                'message': result.get('message', 'Không nhận diện được khuôn mặt')
            })
            
    except Exception as e:
        print(f"Error in process_face_api: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Lỗi server: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Tạo thư mục uploads nếu chưa có
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Khởi tạo database
    from database.database import DatabaseManager
    db_manager = DatabaseManager()
    
    print("Starting Flask-SocketIO server...")
    print("Access laptop interface at: http://localhost:5000")
    print("Access mobile interface at: http://localhost:5000/mobile")
    print("Simple mobile interface at: http://localhost:5000/mobile/simple")
    print("")
    print("⚠️  Lưu ý cho mobile:")
    print("- Camera cần HTTPS để hoạt động trên điện thoại")
    print("- Nếu truy cập từ điện thoại khác, dùng IP: http://[YOUR_IP]:5000/mobile/simple")
    print("- Hoặc cài đặt HTTPS certificate để dùng https://")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
