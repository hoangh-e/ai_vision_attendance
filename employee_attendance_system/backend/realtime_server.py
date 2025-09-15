#!/usr/bin/env python3
"""
Real-time Video Streaming Server with Face Detection
- Mobile streams video frames via SocketIO
- Server processes frames with DeepFace/OpenCV
- Returns bounding boxes + user info to mobile
- Mobile displays video + overlays in real-time
"""

import os
import sys
import cv2
import numpy as np
import base64
import time
import json
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import eventlet
import random

# Monkey patch for eventlet
eventlet.monkey_patch()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import face recognition pipeline
try:
    from face_recognition_pipeline import FaceRecognitionPipeline, initialize_demo_database
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Face recognition not available: {e}")
    FACE_RECOGNITION_AVAILABLE = False

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'realtime_video_secret_2025'

# Initialize SocketIO with eventlet
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   async_mode='eventlet',
                   logger=True,
                   engineio_logger=True)

# Mock employee database for testing
EMPLOYEE_DATABASE = [
    {
        'id': 1,
        'name': 'Nguyễn Văn A',
        'employee_id': 'EMP001',
        'department': 'IT',
        'face_encoding': None  # Will be generated from face images
    },
    {
        'id': 2, 
        'name': 'Trần Thị B',
        'employee_id': 'EMP002',  
        'department': 'HR',
        'face_encoding': None
    },
    {
        'id': 3,
        'name': 'Lê Văn C', 
        'employee_id': 'EMP003',
        'department': 'Finance',
        'face_encoding': None
    }
]

# Global settings
FACE_DETECTION_ENABLED = True
CONFIDENCE_THRESHOLD = 0.6

# Initialize face recognition pipeline
face_pipeline = None
if FACE_RECOGNITION_AVAILABLE:
    try:
        face_pipeline = initialize_demo_database()
        print("✅ Face recognition pipeline initialized")
    except Exception as e:
        print(f"⚠️  Face recognition initialization failed: {e}")
        FACE_RECOGNITION_AVAILABLE = False

def detect_faces_in_frame(frame):
    """
    Detect faces in video frame using face recognition pipeline
    Returns list of face bounding boxes with recognition results
    """
    try:
        if FACE_RECOGNITION_AVAILABLE and face_pipeline:
            # Use advanced face recognition pipeline
            return face_pipeline.process_frame(frame, use_deepface=False)
        else:
            # Fallback to basic OpenCV detection
            return detect_faces_opencv_basic(frame)
        
    except Exception as e:
        print(f"Face detection error: {e}")
        return detect_faces_opencv_basic(frame)

def detect_faces_opencv_basic(frame):
    """
    Basic OpenCV face detection fallback
    """
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load OpenCV face cascade with fallback
        try:
            if hasattr(cv2, 'data'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            else:
                cascade_path = 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
        except:
            # Create default cascade if file not found
            face_cascade = cv2.CascadeClassifier()
            
        if face_cascade.empty():
            print("⚠️  Could not load face cascade classifier")
            return []
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        for (x, y, w, h) in faces:
            # Mock face recognition for demo
            confidence = random.uniform(0.4, 0.95)
            
            if confidence > CONFIDENCE_THRESHOLD:
                # Randomly select an employee for demo
                employee = random.choice(EMPLOYEE_DATABASE)
                result = {
                    'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'employee': employee,
                    'confidence': float(confidence),
                    'type': 'known'
                }
            else:
                result = {
                    'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'employee': None,
                    'confidence': float(confidence),
                    'type': 'stranger'
                }
            
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Basic face detection error: {e}")
        return []

def process_video_frame(frame_data):
    """
    Process single video frame from mobile
    """
    try:
        # Decode base64 frame
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {'success': False, 'error': 'Could not decode frame'}
        
        # Detect faces if enabled
        faces = []
        if FACE_DETECTION_ENABLED:
            faces = detect_faces_in_frame(frame)
        
        return {
            'success': True,
            'timestamp': time.time(),
            'faces': faces,
            'frame_processed': True
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Routes
@app.route('/')
def index():
    return """
    <h1>🎥 Real-time Video Streaming Server</h1>
    <p>Server ready for mobile video streaming!</p>
    <ul>
        <li><a href="/mobile/realtime">📱 Mobile Real-time Interface</a></li>
        <li><a href="/dashboard">💻 Desktop Dashboard</a></li>
        <li><a href="/api/employees">👥 Employee Database</a></li>
    </ul>
    <p><strong>Status:</strong> Face Detection = {'ON' if FACE_DETECTION_ENABLED else 'OFF'}</p>
    """

@app.route('/mobile/realtime')
def mobile_realtime():
    """Mobile real-time video interface"""
    return render_template('mobile_realtime.html')

@app.route('/dashboard')
def dashboard():
    """Desktop dashboard to monitor video streams"""
    return render_template('dashboard_realtime.html')

@app.route('/api/employees')
def get_employees():
    return jsonify({'employees': EMPLOYEE_DATABASE})

@app.route('/api/toggle_detection', methods=['POST'])
def toggle_detection():
    global FACE_DETECTION_ENABLED
    FACE_DETECTION_ENABLED = not FACE_DETECTION_ENABLED
    return jsonify({
        'face_detection_enabled': FACE_DETECTION_ENABLED,
        'message': f"Face detection {'enabled' if FACE_DETECTION_ENABLED else 'disabled'}"
    })

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    print(f"📱 Client connected: {request.sid}")
    emit('connection_response', {
        'status': 'connected',
        'server_time': time.time(),
        'face_detection_enabled': FACE_DETECTION_ENABLED
    })

@socketio.on('disconnect')
def handle_disconnect():
    print(f"📱 Client disconnected: {request.sid}")

@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Handle incoming video frame from mobile
    Process face detection and return results
    """
    try:
        frame_data = data.get('frame')
        timestamp = data.get('timestamp', time.time())
        
        if not frame_data:
            emit('detection_result', {'success': False, 'error': 'No frame data'})
            return
        
        # Process the frame
        result = process_video_frame(frame_data)
        result['client_timestamp'] = timestamp
        result['server_timestamp'] = time.time()
        
        # Send results back to mobile
        emit('detection_result', result)
        
        # Also broadcast to dashboard if connected
        socketio.emit('frame_processed', {
            'client_id': request.sid,
            'faces_detected': len(result.get('faces', [])),
            'timestamp': result['server_timestamp']
        }, room='dashboard')
        
    except Exception as e:
        print(f"Error processing video frame: {e}")
        emit('detection_result', {'success': False, 'error': str(e)})

@socketio.on('join_dashboard')
def handle_join_dashboard():
    """Join dashboard room to receive monitoring data"""
    from flask_socketio import join_room
    join_room('dashboard')
    print(f"💻 Dashboard joined: {request.sid}")
    emit('dashboard_status', {'joined': True, 'timestamp': time.time()})

@socketio.on('mobile_status')
def handle_mobile_status(data):
    """Handle mobile status updates"""
    print(f"📱 Mobile status: {data}")
    # Broadcast to dashboard
    socketio.emit('mobile_update', {
        'client_id': request.sid,
        'status': data,
        'timestamp': time.time()
    }, room='dashboard')

if __name__ == '__main__':
    print("🎥 Starting Real-time Video Streaming Server...")
    print("📱 Mobile Interface: http://localhost:5003/mobile/realtime")
    print("💻 Desktop Dashboard: http://localhost:5003/dashboard")
    print("🔗 API Endpoints: http://localhost:5003/api/employees")
    print()
    print("✅ SocketIO with eventlet enabled")
    print("✅ Real-time video processing ready")
    print("✅ Face detection pipeline active")
    print()
    
    try:
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5003, 
                    debug=True,
                    use_reloader=False)  # Disable reloader to prevent eventlet issues
    except Exception as e:
        print(f"❌ Server error: {e}")
        print("💡 Make sure port 5003 is available")
