#!/usr/bin/env python3
"""
app_unified.py - BHK TECH ATTENDANCE SYSTEM
Unified Flask + SocketIO server
Real-time video streaming from mobile to desktop
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room
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

# =====================================================
# BASIC CONFIGURATION
# =====================================================
app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

# Basic configuration
app.config.update({
    'SECRET_KEY': 'BHK-Tech-Attendance-System-2024',
    'UPLOAD_FOLDER': '../frontend/static/uploads',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024  # 16MB
})

# Initialize SocketIO
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='threading'
)

# =====================================================
# GLOBAL STATE
# =====================================================
app_state = {
    'detection_active': False,
    'connected_clients': {},
    'stats': {
        'total_frames': 0,
        'total_detections': 0,
        'fps': 0,
        'uptime': time.time()
    }
}

# =====================================================
# BASIC SERVICES (FALLBACK IMPLEMENTATION)
# =====================================================
class BasicFaceService:
    """Basic face service for demonstration"""
    def recognize_face(self, frame):
        # Mock implementation - returns empty list
        return []
    
    def detect_and_recognize_faces(self, frame):
        # Mock implementation for enhanced interface
        return []
    
    def save_image_and_vector(self, file, employee_id):
        class MockResult:
            id = 1
        return MockResult()

class BasicEmployeeService:
    """Basic employee service for demonstration"""
    def get_all_employees(self):
        # Mock data
        return [
            {
                'id': 1, 
                'name': 'Demo Employee', 
                'employee_code': 'EMP001',
                'department': 'IT',
                'position': 'Developer',
                'email': 'demo@company.com',
                'phone': '0123456789'
            }
        ]
    
    def create_employee(self, **kwargs):
        class MockEmployee:
            id = 1
        return MockEmployee()
    
    def get_employee_image_count(self, employee_id):
        return 0

# Initialize services
face_service = BasicFaceService()
employee_service = BasicEmployeeService()
print("✅ Basic services initialized")

# =====================================================
# ENHANCED STREAM SERVICE
# =====================================================
try:
    from services.stream_service import StreamService
    # Initialize enhanced stream service
    stream_service = StreamService(socketio, face_service)
    print("✅ Enhanced stream service initialized")
except ImportError as e:
    print(f"⚠️  Could not import enhanced stream service: {e}")
    print("🔄 Using basic fallback stream service")
    
    # Fallback basic service
    class BasicStreamService:
        def __init__(self):
            self.clients = {}
            self.processing_active = False
            self.detection_active = False
        
        def register_client(self, client_id, socket_id, client_info=None):
            self.clients[client_id] = socket_id
            return len(self.clients)
        
        def unregister_client(self, client_id):
            if client_id in self.clients:
                del self.clients[client_id]
            return True
        
        def process_video_frame(self, client_id, frame_data, metadata=None):
            # Basic frame processing
            return {
                'success': True,
                'timestamp': time.time(),
                'faces': [],  # No faces detected in basic mode
                'detection_active': self.detection_active,
                'processing_time': 0.001
            }
        
        def toggle_detection(self):
            self.detection_active = not self.detection_active
            return self.detection_active
        
        def set_detection_active(self, active):
            self.detection_active = active
            return self.detection_active
        
        def get_current_stats(self):
            return {
                'clients': {'total_clients': len(self.clients)},
                'processing': {'detection_active': self.detection_active, 'current_fps': 0},
                'system': {'uptime': time.time() - app_state['stats']['uptime']}
            }
    
    stream_service = BasicStreamService()

# =====================================================
# WEB ROUTES
# =====================================================

@app.route('/')
def index():
    """Desktop interface"""
    return render_template('index.html')

@app.route('/mobile')
def mobile():
    """Mobile interface"""
    return render_template('mobile.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'uptime': time.time() - app_state['stats']['uptime'],
        'stats': app_state['stats'],
        'timestamp': time.time()
    })

# =====================================================
# API ROUTES - Employee Management
# =====================================================

@app.route('/api/employees', methods=['GET'])
def get_employees():
    """Get all employees"""
    try:
        employees = employee_service.get_all_employees()
        return jsonify({'success': True, 'data': employees})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/employees', methods=['POST'])
def create_employee():
    """Create new employee"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
            
        employee = employee_service.create_employee(
            name=data.get('name'),
            employee_code=data.get('employee_code'),
            department=data.get('department'),
            position=data.get('position'),
            email=data.get('email'),
            phone=data.get('phone')
        )
        return jsonify({'success': True, 'data': {'id': employee.id}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/employees/<int:employee_id>', methods=['GET'])
def get_employee(employee_id):
    """Get specific employee"""
    try:
        # Mock implementation
        return jsonify({'success': True, 'data': {'id': employee_id, 'name': 'Demo Employee'}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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
        return jsonify({'success': True, 'data': {'id': result.id}})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# =====================================================
# SOCKETIO EVENT HANDLERS
# =====================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    session['connected_at'] = time.time()
    emit('connection_response', {
        'status': 'connected',
        'server_time': time.time(),
        'detection_active': app_state['detection_active']
    })
    
    # Update client count
    client_count = len(app_state['connected_clients']) + 1
    
    # Broadcast update
    socketio.emit('client_count_update', {
        'count': client_count
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = session.get('client_id')
    if client_id and client_id in app_state['connected_clients']:
        del app_state['connected_clients'][client_id]
        
        # Update client count
        client_count = len(app_state['connected_clients'])
        
        # Broadcast update
        socketio.emit('client_count_update', {
            'count': client_count
        })

@socketio.on('register_mobile_client')
def handle_register_mobile(data):
    """Register mobile client for streaming"""
    client_id = session.get('client_id', str(uuid.uuid4()))
    session['client_id'] = client_id
    # Get socket ID from session or generate new one
    socket_id = session.get('socket_id', str(uuid.uuid4()))
    session['socket_id'] = socket_id
    
    client_info = {
        'type': 'mobile_video_streamer',
        'user_agent': data.get('user_agent', ''),
        'screen_size': data.get('screen_size', ''),
        'timestamp': data.get('timestamp', time.time())
    }
    
    # Register with enhanced stream service
    try:
        client_count = stream_service.register_client(client_id, socket_id, client_info)
        detection_active = stream_service.detection_active
        
        # Also update app state for backwards compatibility
        app_state['connected_clients'][client_id] = client_info
        app_state['detection_active'] = detection_active
        
    except Exception as e:
        print(f"❌ Stream service registration failed: {e}")
        # Fallback to basic registration
        app_state['connected_clients'][client_id] = client_info
        client_count = len(app_state['connected_clients'])
        detection_active = app_state['detection_active']
        
        emit('mobile_registered', {
            'client_id': client_id,
            'detection_active': detection_active
        })
    
    # Broadcast to desktop monitors
    socketio.emit('mobile_connected', {
        'client_id': client_id,
        'client_count': client_count
    })
    
    print(f'📱 Mobile client registered: {client_id}')

@socketio.on('video_frame')
def handle_video_frame(data):
    """Process video frame from mobile using enhanced stream service"""
    try:
        client_id = session.get('client_id', 'unknown')
        
        frame_data = data.get('frame')
        if not frame_data:
            emit('detection_result', {'success': False, 'error': 'No frame data'})
            return
        
        # Use enhanced stream service if available
        try:
            metadata = {
                'resolution': data.get('resolution', 'unknown'),
                'quality': data.get('quality', 0.8),
                'timestamp': data.get('timestamp', time.time())
            }
            
            result = stream_service.process_video_frame(client_id, frame_data, metadata)
            
            # Send result back to mobile
            emit('detection_result', result)
            
            # Update stats for basic compatibility
            if result.get('success'):
                app_state['stats']['total_frames'] += 1
                if result.get('faces'):
                    app_state['stats']['total_detections'] += len(result['faces'])
            
        except Exception as stream_error:
            print(f"⚠️ Enhanced stream service error: {stream_error}")
            print("🔄 Falling back to basic processing")
            
            # Fallback to basic processing
            result = _basic_frame_processing(client_id, frame_data, data)
            emit('detection_result', result)
        
        # Broadcast to desktop monitors
        socketio.emit('frame_processed', {
            'client_id': client_id,
            'detections_count': len(result.get('faces', [])),
            'timestamp': time.time(),
            'stats': app_state['stats']
        })
        
    except Exception as e:
        print(f"❌ Video frame processing error: {e}")
        emit('detection_result', {'success': False, 'error': str(e)})

def _basic_frame_processing(client_id, frame_data, data):
    """Basic frame processing fallback"""
    # Update client stats
    if client_id in app_state['connected_clients']:
        app_state['connected_clients'][client_id]['frames_sent'] += 1
    
    # Update global stats
    app_state['stats']['total_frames'] += 1
    
    result = {
        'success': True,
        'timestamp': time.time(),
        'client_timestamp': data.get('timestamp'),
        'faces': [],  # Using 'faces' key for compatibility
        'detection_active': app_state['detection_active'],
        'frame_count': app_state['stats']['total_frames']
    }
    
    # Basic face detection if active
    if app_state['detection_active']:
        try:
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Load face cascade classifier
                cascade_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')
                if not os.path.exists(cascade_path):
                    # Alternative path for different OpenCV installations
                    cascade_path = 'haarcascade_frontalface_default.xml'
                face_cascade = cv2.CascadeClassifier(cascade_path)
                
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                detections = []
                for (x, y, w, h) in faces:
                    detections.append({
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'confidence': 0.8,
                        'employee': None  # No recognition in basic mode
                    })
                
                result['faces'] = detections
                app_state['stats']['total_detections'] += len(detections)
                
        except Exception as e:
            print(f"❌ Basic face detection error: {e}")
    
    return result

@socketio.on('join_desktop_monitor')
def handle_join_desktop():
    """Desktop joins monitoring room"""
    join_room('desktop_monitors')
    
    emit('desktop_monitor_joined', {
        'stats': app_state['stats'],
        'connected_clients': len(app_state['connected_clients']),
        'detection_active': app_state['detection_active']
    })

@socketio.on('toggle_detection')
def handle_toggle_detection(data):
    """Toggle face detection on/off using enhanced stream service"""
    try:
        # Use enhanced stream service
        detection_active = stream_service.set_detection_active(data.get('active', False))
        
        # Update app state for compatibility
        app_state['detection_active'] = detection_active
        
        socketio.emit('detection_status_changed', {
            'active': detection_active,
            'timestamp': time.time()
        })
        
        print(f"🔍 Face detection: {'ENABLED' if detection_active else 'DISABLED'}")
        
    except Exception as e:
        print(f"❌ Toggle detection error: {e}")
        # Fallback to basic toggle
        app_state['detection_active'] = data.get('active', False)
        
        socketio.emit('detection_status_changed', {
            'active': app_state['detection_active'],
            'timestamp': time.time()
        })

@socketio.on('get_stats')
def handle_get_stats():
    """Get current system stats using enhanced stream service"""
    try:
        # Use enhanced stream service stats
        enhanced_stats = stream_service.get_current_stats()
        
        emit('stats_update', {
            'stats': enhanced_stats,
            'timestamp': time.time()
        })
        
    except Exception as e:
        print(f"❌ Get stats error: {e}")
        # Fallback to basic stats
        stats = app_state['stats'].copy()
        stats['connected_clients'] = len(app_state['connected_clients'])
        stats['detection_active'] = app_state['detection_active']
        
        emit('stats_update', {
            'stats': stats,
            'timestamp': time.time()
        })

@socketio.on('ping')
def handle_ping(data):
    """Handle ping for latency measurement"""
    emit('pong', data)

# =====================================================
# ERROR HANDLERS
# =====================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == '__main__':
    # Create required directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("🚀 STARTING BHK TECH ATTENDANCE SYSTEM")
    print("=" * 60)
    print(f"🖥️  Desktop Interface: http://localhost:5000")
    print(f"📱 Mobile Interface:   http://[YOUR_IP]:5000/mobile")
    print(f"🔧 API Endpoints:      http://localhost:5000/api/")
    print(f"❤️  Health Check:      http://localhost:5000/health")
    print("=" * 60)
    print("✅ Real-time video streaming ready")
    print("✅ Basic face detection (OpenCV)")
    print("✅ SocketIO enabled")
    print("=" * 60)
    print("💡 To get your computer's IP address:")
    print("   🪟 Windows: ipconfig")
    print("   🍎 Mac/Linux: ifconfig")
    print(f"📱 On mobile, visit: http://[YOUR_IP]:5000/mobile")
    print()
    print("🎯 FEATURES:")
    print("   • Real-time mobile camera streaming")
    print("   • Desktop monitoring dashboard")
    print("   • Basic face detection")
    print("   • Employee management API")
    print("   • Live statistics")
    print()
    
    # Start the server
    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # Avoid conflicts with SocketIO
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        print("👋 Goodbye!")