#!/usr/bin/env python3
"""
app.py - BHK TECH ATTENDANCE SYSTEM
Enhanced Flask + SocketIO server with mobile-to-desktop video streaming
Real-time face detection and attendance tracking
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
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

# =====================================================
# BASIC CONFIGURATION
# =====================================================
app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

# Enhanced configuration
app.config.update({
    'SECRET_KEY': 'BHK-Tech-Attendance-System-2024-Enhanced',
    'UPLOAD_FOLDER': '../frontend/static/uploads',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024  # 16MB
})

# Initialize SocketIO with enhanced configuration
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
# ENHANCED GLOBAL STATE
# =====================================================
app_state = {
    'detection_active': False,
    'mobile_clients': {},  # Track mobile clients
    'desktop_clients': {},  # Track desktop monitoring clients
    'current_frame': None,  # Store latest frame from mobile
    'frame_timestamp': None,
    'stats': {
        'total_frames': 0,
        'total_detections': 0,
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
# ENHANCED SERVICES WITH OPENCV INTEGRATION
# =====================================================
class EnhancedFaceService:
    """Enhanced face service with OpenCV integration"""
    
    def __init__(self):
        self.face_cascade = None
        self.face_recognizer = None
        self.initialize_opencv()
    
    def initialize_opencv(self):
        """Initialize OpenCV components"""
        try:
            # Load Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                logger.warning("Could not load face cascade classifier")
                self.face_cascade = None
            else:
                logger.info("Face cascade classifier loaded successfully")
                
        except Exception as e:
            logger.error(f"Error initializing OpenCV: {e}")
            self.face_cascade = None
    
    def detect_faces(self, frame):
        """Detect faces in frame using OpenCV"""
        if self.face_cascade is None:
            return []
        
        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return faces.tolist() if len(faces) > 0 else []
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def recognize_face(self, frame):
        """Legacy compatibility method"""
        faces = self.detect_faces(frame)
        return [{'bbox': face, 'confidence': 0.8, 'name': 'Unknown'} for face in faces]
    
    def detect_and_recognize_faces(self, frame):
        """Enhanced method for face detection and recognition"""
        faces = self.detect_faces(frame)
        results = []
        
        for face in faces:
            x, y, w, h = face
            results.append({
                'bbox': [x, y, w, h],
                'confidence': 0.8,
                'name': 'Unknown Employee',
                'employee_id': None,
                'timestamp': datetime.now().isoformat()
            })
        
        return results
    
    def save_image_and_vector(self, file, employee_id):
        """Save employee image with enhanced metadata"""
        try:
            if file and file.filename:
                # Generate unique filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_id = str(uuid.uuid4())[:8]
                filename = f"emp_{employee_id}_{timestamp}_{unique_id}.jpg"
                
                # Ensure upload directory exists
                upload_dir = app.config['UPLOAD_FOLDER']
                os.makedirs(upload_dir, exist_ok=True)
                
                # Save file
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

class EnhancedEmployeeService:
    """Enhanced employee service with better data management"""
    
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
        """Get all employees with enhanced data"""
        return self.employees
    
    def create_employee(self, **kwargs):
        """Create new employee with validation"""
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
    
    def get_employee_image_count(self, employee_id):
        """Get image count for employee"""
        try:
            upload_dir = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_dir):
                return 0
            
            # Count files matching employee pattern
            count = 0
            for filename in os.listdir(upload_dir):
                if filename.startswith(f"emp_{employee_id}_"):
                    count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Error counting employee images: {e}")
            return 0

# Initialize enhanced services
face_service = EnhancedFaceService()
employee_service = EnhancedEmployeeService()

# =====================================================
# FRAME PROCESSING UTILITIES
# =====================================================
def decode_base64_frame(base64_data):
    """Decode base64 frame data to OpenCV format"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return frame
        
    except Exception as e:
        logger.error(f"Error decoding frame: {e}")
        return None

def encode_frame_to_base64(frame):
    """Encode OpenCV frame to base64"""
    try:
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/jpeg;base64,{img_base64}"
        
    except Exception as e:
        logger.error(f"Error encoding frame: {e}")
        return None

def process_frame_with_detection(frame):
    """Process frame with face detection"""
    try:
        if frame is None:
            return None, []
        
        # Detect faces
        detections = face_service.detect_and_recognize_faces(frame)
        
        # Draw detection boxes
        processed_frame = frame.copy()
        for detection in detections:
            x, y, w, h = detection['bbox']
            
            # Draw rectangle
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{detection['name']} ({detection['confidence']:.2f})"
            cv2.putText(processed_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return processed_frame, detections
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return frame, []

# =====================================================
# FLASK ROUTES
# =====================================================
@app.route('/')
def index():
    """Desktop monitoring interface"""
    return render_template('index.html')

@app.route('/mobile')
def mobile():
    """Mobile camera interface"""
    return render_template('mobile.html')

@app.route('/management')
def management():
    """Employee management interface"""
    employees = employee_service.get_all_employees()
    return render_template('management.html', employees=employees)

@app.route('/api/employees', methods=['GET'])
def get_employees():
    """API endpoint for employees"""
    try:
        employees = employee_service.get_all_employees()
        return jsonify({
            'success': True,
            'data': employees,
            'count': len(employees)
        })
    except Exception as e:
        logger.error(f"Error getting employees: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/employees', methods=['POST'])
def create_employee():
    """API endpoint to create employee"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        employee = employee_service.create_employee(**data)
        if employee:
            return jsonify({
                'success': True,
                'data': {'id': employee.id, 'name': data.get('name')},
                'message': 'Employee created successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to create employee'}), 500
            
    except Exception as e:
        logger.error(f"Error creating employee: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload_employee_image/<int:employee_id>', methods=['POST'])
def upload_employee_image(employee_id):
    """Upload employee training image"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Save image
        result = face_service.save_image_and_vector(file, employee_id)
        if result:
            image_count = employee_service.get_employee_image_count(employee_id)
            return jsonify({
                'success': True,
                'data': {'image_id': result.id, 'total_images': image_count},
                'message': 'Image uploaded successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to save image'}), 500
            
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics"""
    try:
        current_time = time.time()
        uptime = current_time - app_state['stats']['uptime']
        
        stats = {
            'detection_active': app_state['detection_active'],
            'mobile_clients': len(app_state['mobile_clients']),
            'desktop_clients': len(app_state['desktop_clients']),
            'total_frames': app_state['stats']['total_frames'],
            'total_detections': app_state['stats']['total_detections'],
            'fps': app_state['stats']['fps'],
            'uptime_seconds': int(uptime),
            'uptime_formatted': f"{int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}",
            'last_frame_time': app_state['stats']['last_frame_time']
        }
        
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
    client_id = request.sid
    logger.info(f"Client connected: {client_id}")
    
    # Send connection confirmation
    emit('connection_status', {
        'connected': True,
        'client_id': client_id,
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    logger.info(f"Client disconnected: {client_id}")
    
    # Remove from tracking
    if client_id in app_state['mobile_clients']:
        del app_state['mobile_clients'][client_id]
        app_state['stats']['mobile_connections'] = len(app_state['mobile_clients'])
        logger.info(f"Removed mobile client: {client_id}")
    
    if client_id in app_state['desktop_clients']:
        del app_state['desktop_clients'][client_id]
        app_state['stats']['desktop_connections'] = len(app_state['desktop_clients'])
        logger.info(f"Removed desktop client: {client_id}")

@socketio.on('register_mobile_client')
def handle_mobile_registration(data):
    """Register mobile client for video streaming"""
    client_id = request.sid
    
    # Store mobile client info
    app_state['mobile_clients'][client_id] = {
        'id': client_id,
        'registered_at': datetime.now().isoformat(),
        'device_info': data.get('device_info', {}),
        'last_frame_time': None
    }
    
    app_state['stats']['mobile_connections'] = len(app_state['mobile_clients'])
    
    logger.info(f"Mobile client registered: {client_id}")
    
    # Join mobile room
    join_room('mobile_clients')
    
    # Confirm registration
    emit('mobile_registration_confirmed', {
        'client_id': client_id,
        'status': 'registered',
        'timestamp': datetime.now().isoformat()
    })
    
    # Notify desktop clients
    socketio.emit('mobile_client_connected', {
        'client_id': client_id,
        'device_info': data.get('device_info', {}),
        'timestamp': datetime.now().isoformat()
    }, room='desktop_monitors')

@socketio.on('register_desktop_monitor')
def handle_desktop_registration(data):
    """Register desktop client for monitoring"""
    client_id = request.sid
    
    # Store desktop client info
    app_state['desktop_clients'][client_id] = {
        'id': client_id,
        'registered_at': datetime.now().isoformat(),
        'monitor_settings': data.get('settings', {}),
        'last_activity': datetime.now().isoformat()
    }
    
    app_state['stats']['desktop_connections'] = len(app_state['desktop_clients'])
    
    logger.info(f"Desktop monitor registered: {client_id}")
    
    # Join desktop room
    join_room('desktop_monitors')
    
    # Confirm registration
    emit('desktop_registration_confirmed', {
        'client_id': client_id,
        'status': 'registered',
        'timestamp': datetime.now().isoformat()
    })
    
    # Send current frame if available
    with frame_lock:
        if app_state['current_frame'] is not None:
            emit('video_frame_update', {
                'frame': app_state['current_frame'],
                'timestamp': app_state['frame_timestamp'],
                'source': 'mobile'
            })

@socketio.on('mobile_frame_received')
def handle_mobile_frame(data):
    """Handle video frame from mobile client - KEY EVENT FOR DESKTOP DISPLAY"""
    client_id = request.sid
    
    try:
        # Validate mobile client
        if client_id not in app_state['mobile_clients']:
            emit('error', {'message': 'Mobile client not registered'})
            return
        
        # Get frame data
        frame_data = data.get('frame')
        if not frame_data:
            emit('error', {'message': 'No frame data received'})
            return
        
        # Update mobile client last frame time
        app_state['mobile_clients'][client_id]['last_frame_time'] = datetime.now().isoformat()
        
        # Decode frame if detection is active
        processed_frame_data = frame_data
        detections = []
        
        if app_state['detection_active']:
            # Decode and process frame
            frame = decode_base64_frame(frame_data)
            if frame is not None:
                processed_frame, detections = process_frame_with_detection(frame)
                if processed_frame is not None:
                    processed_frame_data = encode_frame_to_base64(processed_frame)
                    
                    # Update detection stats
                    app_state['stats']['total_detections'] += len(detections)
        
        # Update global frame state
        with frame_lock:
            app_state['current_frame'] = processed_frame_data
            app_state['frame_timestamp'] = datetime.now().isoformat()
            app_state['stats']['total_frames'] += 1
            app_state['stats']['last_frame_time'] = app_state['frame_timestamp']
        
        # Broadcast to all desktop monitors
        socketio.emit('video_frame_update', {
            'frame': processed_frame_data,
            'timestamp': app_state['frame_timestamp'],
            'source': 'mobile',
            'client_id': client_id,
            'detections': detections,
            'detection_active': app_state['detection_active']
        }, room='desktop_monitors')
        
        # Send confirmation to mobile
        emit('frame_processed', {
            'status': 'success',
            'timestamp': app_state['frame_timestamp'],
            'detections_count': len(detections)
        })
        
        logger.debug(f"Processed frame from mobile {client_id}, {len(detections)} detections")
        
    except Exception as e:
        logger.error(f"Error processing mobile frame: {e}")
        emit('error', {'message': f'Frame processing error: {str(e)}'})

@socketio.on('toggle_detection')
def handle_toggle_detection(data):
    """Toggle face detection on/off"""
    client_id = request.sid
    
    # Toggle detection state
    app_state['detection_active'] = not app_state['detection_active']
    
    logger.info(f"Detection toggled by {client_id}: {app_state['detection_active']}")
    
    # Broadcast state change to all clients
    socketio.emit('detection_state_changed', {
        'detection_active': app_state['detection_active'],
        'timestamp': datetime.now().isoformat(),
        'changed_by': client_id
    })
    
    # Send confirmation
    emit('detection_toggled', {
        'detection_active': app_state['detection_active'],
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('get_system_status')
def handle_get_status(data):
    """Get current system status"""
    try:
        current_time = time.time()
        uptime = current_time - app_state['stats']['uptime']
        
        status = {
            'detection_active': app_state['detection_active'],
            'mobile_clients': len(app_state['mobile_clients']),
            'desktop_clients': len(app_state['desktop_clients']),
            'has_current_frame': app_state['current_frame'] is not None,
            'stats': {
                'total_frames': app_state['stats']['total_frames'],
                'total_detections': app_state['stats']['total_detections'],
                'uptime_seconds': int(uptime),
                'last_frame_time': app_state['stats']['last_frame_time']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        emit('system_status_update', status)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        emit('error', {'message': f'Status error: {str(e)}'})

@socketio.on('ping')
def handle_ping(data):
    """Handle ping for connection testing"""
    emit('pong', {
        'timestamp': datetime.now().isoformat(),
        'server_time': time.time()
    })

# =====================================================
# ERROR HANDLERS
# =====================================================
@app.errorhandler(404)
def page_not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# =====================================================
# APPLICATION STARTUP
# =====================================================
def initialize_application():
    """Initialize application components"""
    try:
        # Ensure upload directory exists
        upload_dir = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_dir, exist_ok=True)
        logger.info(f"Upload directory ready: {upload_dir}")
        
        # Initialize services
        logger.info("Services initialized successfully")
        
        # Log startup info
        logger.info("BHK Tech Attendance System - Enhanced Version")
        logger.info(f"Detection service: {'Active' if face_service.face_cascade else 'Limited'}")
        logger.info(f"Employee count: {len(employee_service.get_all_employees())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        return False

# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == '__main__':
    print("🚀 Starting BHK Tech Attendance System - Enhanced Version")
    print("=" * 60)
    
    # Initialize application
    if not initialize_application():
        print("❌ Failed to initialize application")
        sys.exit(1)
    
    print("✅ Application initialized successfully")
    print(f"📱 Mobile Interface: http://localhost:5000/mobile")
    print(f"🖥️  Desktop Monitor: http://localhost:5000/")
    print(f"👥 Management: http://localhost:5000/management")
    print("=" * 60)
    
    try:
        # Start the server
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for production
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        print("👋 BHK Tech Attendance System shutdown complete")