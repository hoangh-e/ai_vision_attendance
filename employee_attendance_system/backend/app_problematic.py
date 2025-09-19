#!/usr/bin/env python3
"""
app.py - BHK TECH ATTENDANCE SYSTEM - MAIN SERVER
Consolidated Flask + SocketIO server with all functionality
"""

from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import cv2
import base64
import numpy as np
import time
import os
import sys
import logging
import uuid
import threading
from werkzeug.utils import secure_filename
from datetime import datetime
import json

# Add backend to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# FLASK APP CONFIGURATION
# =====================================================
app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

# Consolidated configuration
app.config.update({
    'SECRET_KEY': 'BHK-Tech-Attendance-System-2024',
    'UPLOAD_FOLDER': '../frontend/static/uploads',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
    'CURRENT_TIME': datetime.now().isoformat()
})

# Initialize SocketIO
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    logger=True,
    engineio_logger=True
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================
# GLOBAL STATE
# =====================================================
app_state = {
    'detection_active': False,
    'mobile_clients': {},      # Track mobile streaming clients
    'desktop_clients': {},     # Track desktop monitoring clients
    'current_frame': None,     # Latest frame from mobile
    'frame_timestamp': None,
    'stats': {
        'total_frames': 0,
        'total_detections': 0,
        'total_recognitions': 0,
        'mobile_connections': 0,
        'desktop_connections': 0,
        'fps': 0,
        'uptime': time.time(),
        'last_frame_time': None
    }
}

# Thread lock for frame operations
frame_lock = threading.Lock()

# =====================================================
# INITIALIZE SERVICES
# =====================================================

# Face Service - Try enhanced version first, fallback to basic
try:
    from services.face_service import FaceService
    face_service = FaceService()
    logger.info("✅ Enhanced Face Service loaded")
except ImportError:
    logger.warning("⚠️ Enhanced Face Service not found, using basic implementation")
    
    class BasicFaceService:
        """Basic face service implementation"""
        def __init__(self):
            self.face_cascade = None
            self.initialize_opencv()
        
        def initialize_opencv(self):
            """Initialize OpenCV face detection"""
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("✅ OpenCV face detection initialized")
            except Exception as e:
                logger.error(f"❌ OpenCV initialization failed: {e}")
        
        def detect_faces(self, frame):
            """Basic face detection using OpenCV"""
            if self.face_cascade is None:
                return []
            
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                return [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} 
                       for (x, y, w, h) in faces]
            except Exception as e:
                logger.error(f"Face detection error: {e}")
                return []
        
        def recognize_face(self, frame):
            """Basic face recognition (mock implementation)"""
            faces = self.detect_faces(frame)
            results = []
            
            for face in faces:
                results.append({
                    'face': face,
                    'employee': None,  # No recognition in basic mode
                    'confidence': 0.0
                })
            
            return results
        
        def save_image_and_vector(self, file, employee_id):
            """Save employee image"""
            try:
                filename = secure_filename(file.filename)
                timestamp = int(time.time())
                filename = f"emp_{employee_id}_{timestamp}_{filename}"
                
                upload_dir = app.config['UPLOAD_FOLDER']
                os.makedirs(upload_dir, exist_ok=True)
                
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                
                logger.info(f"Saved employee image: {filename}")
                
                class MockResult:
                    def __init__(self, id_val):
                        self.id = id_val
                
                return MockResult(len(os.listdir(upload_dir)))
                
            except Exception as e:
                logger.error(f"Error saving employee image: {e}")
                return None
        
        def get_employee_image_count(self, employee_id):
            """Get image count for employee"""
            try:
                upload_dir = app.config['UPLOAD_FOLDER']
                if not os.path.exists(upload_dir):
                    return 0
                
                count = 0
                for filename in os.listdir(upload_dir):
                    if filename.startswith(f"emp_{employee_id}_"):
                        count += 1
                
                return count
                
            except Exception as e:
                logger.error(f"Error counting employee images: {e}")
                return 0
    
    face_service = BasicFaceService()

# Employee Service - Try enhanced version first, fallback to basic
try:
    from services.employee_service import EmployeeService
    employee_service = EmployeeService()
    logger.info("✅ Enhanced Employee Service loaded")
except ImportError:
    logger.warning("⚠️ Enhanced Employee Service not found, using basic implementation")
    
    class BasicEmployeeService:
        """Basic employee service implementation"""
        def __init__(self):
            self.employees = [
                {
                    'id': 1, 
                    'name': 'Nguyễn Văn An', 
                    'employee_code': 'EMP001',
                    'department': 'IT',
                    'position': 'Developer',
                    'email': 'an.nguyen@company.com',
                    'phone': '0901234567',
                    'status': 'active',
                    'created_at': '2024-01-01'
                },
                {
                    'id': 2, 
                    'name': 'Trần Thị Bình', 
                    'employee_code': 'EMP002',
                    'department': 'HR',
                    'position': 'Manager',
                    'email': 'binh.tran@company.com',
                    'phone': '0901234568',
                    'status': 'active',
                    'created_at': '2024-01-02'
                }
            ]
        
        def get_all_employees(self):
            """Get all employees with image count"""
            employees = self.employees.copy()
            for emp in employees:
                emp['image_count'] = face_service.get_employee_image_count(emp['id'])
            return employees
        
        def create_employee(self, **kwargs):
            """Create new employee"""
            try:
                new_id = max([emp['id'] for emp in self.employees]) + 1 if self.employees else 1
                
                new_employee = {
                    'id': new_id,
                    'name': kwargs.get('name', 'Unknown'),
                    'employee_code': kwargs.get('employee_code', f'EMP{new_id:03d}'),
                    'department': kwargs.get('department', 'General'),
                    'position': kwargs.get('position', 'Employee'),
                    'email': kwargs.get('email', ''),
                    'phone': kwargs.get('phone', ''),
                    'status': 'active',
                    'created_at': datetime.now().strftime('%Y-%m-%d')
                }
                
                self.employees.append(new_employee)
                logger.info(f"Created new employee: {new_employee['name']}")
                
                class MockEmployee:
                    def __init__(self, emp_data):
                        self.id = emp_data['id']
                        self.__dict__.update(emp_data)
                
                return MockEmployee(new_employee)
                
            except Exception as e:
                logger.error(f"Error creating employee: {e}")
                return None
        
        def delete_employee(self, employee_id):
            """Delete employee"""
            original_length = len(self.employees)
            self.employees = [emp for emp in self.employees if emp['id'] != employee_id]
            return len(self.employees) < original_length
    
    employee_service = BasicEmployeeService()

# =====================================================
# FRAME PROCESSING UTILITIES
# =====================================================
def decode_base64_frame(base64_data):
    """Decode base64 frame data to OpenCV format"""
    try:
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        img_data = base64.b64decode(base64_data)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return frame
        
    except Exception as e:
        logger.error(f"Error decoding frame: {e}")
        return None

def encode_frame_to_base64(frame):
    """Encode OpenCV frame to base64"""
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
        
    except Exception as e:
        logger.error(f"Error encoding frame: {e}")
        return None

# =====================================================
# WEB ROUTES
# =====================================================

@app.route('/')
def index():
    """Desktop monitoring interface"""
    return render_template('index.html')

@app.route('/mobile')
def mobile():
    """Mobile camera interface"""
    return render_template('mobile.html')

# =====================================================
# API ENDPOINTS
# =====================================================

@app.route('/api/employees', methods=['GET'])
def get_employees():
    """Get all employees"""
    try:
        employees = employee_service.get_all_employees()
        return jsonify({'success': True, 'data': employees})
    except Exception as e:
        logger.error(f"Error getting employees: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/employees', methods=['POST'])
def create_employee():
    """Create new employee"""
    try:
        data = request.get_json()
        employee = employee_service.create_employee(**data)
        
        if employee:
            return jsonify({'success': True, 'data': {'id': employee.id}})
        else:
            return jsonify({'success': False, 'error': 'Failed to create employee'}), 500
            
    except Exception as e:
        logger.error(f"Error creating employee: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/employees/<int:employee_id>', methods=['DELETE'])
def delete_employee(employee_id):
    """Delete employee"""
    try:
        success = employee_service.delete_employee(employee_id)
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Employee not found'}), 404
            
    except Exception as e:
        logger.error(f"Error deleting employee: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload-face', methods=['POST'])
def upload_face():
    """Upload face image for employee"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        employee_id = request.form.get('employee_id')
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        result = face_service.save_image_and_vector(file, employee_id)
        
        if result:
            return jsonify({'success': True, 'data': {'id': result.id}})
        else:
            return jsonify({'success': False, 'error': 'Failed to save image'}), 500
    
    except Exception as e:
        logger.error(f"Error uploading face: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        current_time = time.time()
        uptime = current_time - app_state['stats']['uptime']
        
        stats = app_state['stats'].copy()
        stats['uptime_seconds'] = int(uptime)
        stats['mobile_clients'] = len(app_state['mobile_clients'])
        stats['desktop_clients'] = len(app_state['desktop_clients'])
        stats['detection_active'] = app_state['detection_active']
        
        return jsonify({'success': True, 'data': stats})
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# =====================================================
# SOCKETIO EVENT HANDLERS
# =====================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    session['connected_at'] = time.time()
    session['client_id'] = str(uuid.uuid4())
    
    emit('connection_response', {
        'status': 'connected',
        'server_time': time.time(),
        'detection_active': app_state['detection_active'],
        'client_id': session['client_id']
    })
    
    logger.info(f"Client connected: {session['client_id']}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = session.get('client_id')
    
    if client_id:
        # Remove from tracking
        app_state['mobile_clients'].pop(client_id, None)
        app_state['desktop_clients'].pop(client_id, None)
        
        # Update stats
        app_state['stats']['mobile_connections'] = len(app_state['mobile_clients'])
        app_state['stats']['desktop_connections'] = len(app_state['desktop_clients'])
        
        # Broadcast update
        socketio.emit('client_count_update', {
            'mobile_count': len(app_state['mobile_clients']),
            'desktop_count': len(app_state['desktop_clients'])
        })
        
        logger.info(f"Client disconnected: {client_id}")

@socketio.on('register_mobile_client')
def handle_register_mobile(data):
    """Register mobile client for streaming"""
    client_id = session.get('client_id')
    
    if client_id:
        client_info = {
            'type': 'mobile_video_streamer',
            'user_agent': data.get('user_agent', ''),
            'screen_size': data.get('screen_size', ''),
            'timestamp': time.time()
        }
        
        app_state['mobile_clients'][client_id] = client_info
        app_state['stats']['mobile_connections'] = len(app_state['mobile_clients'])
        
        emit('mobile_registered', {
            'client_id': client_id,
            'detection_active': app_state['detection_active']
        })
        
        # Broadcast to desktop monitors
        socketio.emit('mobile_connected', {
            'client_id': client_id,
            'mobile_count': len(app_state['mobile_clients'])
        })
        
        logger.info(f'📱 Mobile client registered: {client_id}')

@socketio.on('register_desktop_client')
def handle_register_desktop(data):
    """Register desktop client for monitoring"""
    client_id = session.get('client_id')
    
    if client_id:
        client_info = {
            'type': 'desktop_monitor',
            'user_agent': data.get('user_agent', ''),
            'timestamp': time.time()
        }
        
        app_state['desktop_clients'][client_id] = client_info
        app_state['stats']['desktop_connections'] = len(app_state['desktop_clients'])
        
        emit('desktop_registered', {
            'client_id': client_id,
            'detection_active': app_state['detection_active']
        })
        
        logger.info(f'🖥️ Desktop client registered: {client_id}')

@socketio.on('video_frame')
def handle_video_frame(data):
    """Process video frame from mobile"""
    try:
        client_id = session.get('client_id', 'unknown')
        frame_data = data.get('frame')
        
        if not frame_data:
            emit('detection_result', {'success': False, 'error': 'No frame data'})
            return
        
        # Decode frame
        frame = decode_base64_frame(frame_data)
        if frame is None:
            emit('detection_result', {'success': False, 'error': 'Invalid frame data'})
            return
        
        # Update stats
        app_state['stats']['total_frames'] += 1
        app_state['stats']['last_frame_time'] = time.time()
        
        # Store current frame for desktop monitoring
        with frame_lock:
            app_state['current_frame'] = frame.copy()
            app_state['frame_timestamp'] = time.time()
        
        # Process frame if detection is active
        detection_results = []
        if app_state['detection_active']:
            try:
                detection_results = face_service.recognize_face(frame)
                app_state['stats']['total_detections'] += len(detection_results)
            except Exception as e:
                logger.error(f"Face recognition error: {e}")
        
        # Prepare response
        response = {
            'success': True,
            'faces': detection_results,
            'timestamp': time.time(),
            'frame_count': app_state['stats']['total_frames']
        }
        
        # Send result back to mobile
        emit('detection_result', response)
        
        # Broadcast frame to desktop monitors
        frame_encoded = encode_frame_to_base64(frame)
        if frame_encoded:
            socketio.emit('video_frame_update', {
                'frame': frame_encoded,
                'faces': detection_results,
                'timestamp': time.time(),
                'client_id': client_id
            }, room=None)  # Broadcast to all
        
    except Exception as e:
        logger.error(f"Error processing video frame: {e}")
        emit('detection_result', {'success': False, 'error': str(e)})

@socketio.on('toggle_detection')
def handle_toggle_detection(data):
    """Toggle face detection on/off"""
    try:
        new_state = data.get('active', False)
        app_state['detection_active'] = new_state
        
        # Broadcast state change to all clients
        socketio.emit('detection_state_changed', {
            'active': new_state,
            'timestamp': time.time()
        })
        
        logger.info(f"Detection {'activated' if new_state else 'deactivated'}")
        
    except Exception as e:
        logger.error(f"Error toggling detection: {e}")

@socketio.on('get_current_frame')
def handle_get_current_frame():
    """Send current frame to requesting client"""
    try:
        with frame_lock:
            if app_state['current_frame'] is not None:
                frame_encoded = encode_frame_to_base64(app_state['current_frame'])
                if frame_encoded:
                    emit('current_frame_update', {
                        'frame': frame_encoded,
                        'timestamp': app_state['frame_timestamp']
                    })
    except Exception as e:
        logger.error(f"Error sending current frame: {e}")

# =====================================================
# ERROR HANDLERS
# =====================================================

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'success': False, 'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# =====================================================
# APPLICATION STARTUP
# =====================================================

def initialize_app():
    """Initialize application components"""
    try:
        # Create upload directory
        upload_dir = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_dir, exist_ok=True)
        logger.info(f"✅ Upload directory ready: {upload_dir}")
        
        # Initialize services (already done above)
        logger.info("✅ Services initialized")
        
        # Set startup time
        app_state['stats']['uptime'] = time.time()
        
        logger.info("🚀 Application initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Application initialization failed: {e}")
        raise

# =====================================================
# MAIN ENTRY POINT
# =====================================================

if __name__ == '__main__':
    try:
        # Initialize application
        initialize_app()
        
        # Start the server
        logger.info("🚀 Starting BHK Tech Attendance System")
        logger.info("📱 Mobile Interface: http://localhost:5000/mobile")
        logger.info("🖥️  Desktop Monitor: http://localhost:5000/")
        
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True
        )
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)