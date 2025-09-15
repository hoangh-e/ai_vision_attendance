#!/usr/bin/env python3
"""
main_app.py - BHK TECH ATTENDANCE SYSTEM
Unified Flask + SocketIO server on port 5000
Real-time video streaming from mobile to desktop
"""

from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room
import cv2
import base64
import numpy as np
import time
import os
import sys
import logging
import uuid

# =====================================================
# BASIC CONFIGURATION
# =====================================================
app = Flask(__name__, 
            template_folder='frontend/templates',
            static_folder='frontend/static')

# Basic configuration
app.config.update({
    'SECRET_KEY': 'BHK-Tech-Attendance-System-2024',
    'UPLOAD_FOLDER': 'frontend/static/uploads',
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
# BASIC SERVICES (MOCK IMPLEMENTATION)
# =====================================================
class BasicFaceService:
    """Basic face service for demonstration"""
    def recognize_face(self, frame):
        # Mock implementation - returns empty list
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

# Initialize basic services
face_service = BasicFaceService()
employee_service = BasicEmployeeService()

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

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    stats = app_state['stats'].copy()
    stats['connected_clients'] = len(app_state['connected_clients'])
    stats['detection_active'] = app_state['detection_active']
    return jsonify({'success': True, 'data': stats})

# =====================================================
# SOCKET.IO EVENTS
# =====================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = str(uuid.uuid4())
    session['client_id'] = client_id
    print(f'✅ Client connected: {client_id}')
    
    emit('server_ready', {
        'status': 'connected',
        'client_id': client_id,
        'server_time': time.time(),
        'detection_active': app_state['detection_active']
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = session.get('client_id', 'unknown')
    print(f'❌ Client disconnected: {client_id}')
    
    # Remove from app state
    if client_id in app_state['connected_clients']:
        del app_state['connected_clients'][client_id]
    
    # Broadcast update
    socketio.emit('client_count_update', {
        'count': len(app_state['connected_clients'])
    })

@socketio.on('register_mobile_client')
def handle_register_mobile(data):
    """Register mobile client for streaming"""
    client_id = session.get('client_id', str(uuid.uuid4()))
    session['client_id'] = client_id
    
    client_info = {
        'type': 'mobile',
        'user_agent': data.get('user_agent', ''),
        'connected_at': time.time(),
        'frames_sent': 0
    }
    
    # Add to app state
    app_state['connected_clients'][client_id] = client_info
    
    emit('mobile_registered', {
        'client_id': client_id,
        'detection_active': app_state['detection_active']
    })
    
    # Broadcast to desktop monitors
    socketio.emit('mobile_connected', {
        'client_id': client_id,
        'client_count': len(app_state['connected_clients'])
    })
    
    print(f'📱 Mobile client registered: {client_id}')

@socketio.on('video_frame')
def handle_video_frame(data):
    """Process video frame from mobile"""
    try:
        client_id = session.get('client_id', 'unknown')
        
        frame_data = data.get('frame')
        if not frame_data:
            emit('detection_result', {'success': False, 'error': 'No frame data'})
            return
        
        # Update client stats
        if client_id in app_state['connected_clients']:
            app_state['connected_clients'][client_id]['frames_sent'] += 1
        
        # Update global stats
        app_state['stats']['total_frames'] += 1
        
        # Basic result structure
        result = {
            'success': True,
            'timestamp': time.time(),
            'client_timestamp': data.get('timestamp'),
            'detections': [],
            'detection_active': app_state['detection_active'],
            'frame_count': app_state['stats']['total_frames']
        }
        
        # Face detection if active
        if app_state['detection_active']:
            try:
                # Decode frame for face detection
                if ',' in frame_data:
                    frame_data = frame_data.split(',')[1]
                
                img_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Basic OpenCV face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    detections = []
                    for (x, y, w, h) in faces:
                        detections.append({
                            'name': 'Unknown Face',
                            'confidence': 0.8,
                            'bbox': [int(x), int(y), int(w), int(h)]
                        })
                    
                    result['detections'] = detections
                    app_state['stats']['total_detections'] += len(detections)
                    
            except Exception as e:
                print(f"❌ Face detection error: {e}")
        
        # Send result back to mobile
        emit('detection_result', result)
        
        # Broadcast to desktop monitors
        socketio.emit('frame_processed', {
            'client_id': client_id,
            'detections_count': len(result.get('detections', [])),
            'timestamp': time.time(),
            'total_frames': app_state['stats']['total_frames']
        })
        
    except Exception as e:
        print(f"❌ Video frame processing error: {e}")
        emit('detection_result', {'success': False, 'error': str(e)})

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
    """Toggle face detection on/off"""
    app_state['detection_active'] = data.get('active', False)
    
    socketio.emit('detection_status_changed', {
        'active': app_state['detection_active'],
        'timestamp': time.time()
    })
    
    print(f"🔍 Face detection: {'ENABLED' if app_state['detection_active'] else 'DISABLED'}")

@socketio.on('get_stats')
def handle_get_stats():
    """Get current system stats"""
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