#!/usr/bin/env python3
"""
main.py - BHK TECH ATTENDANCE SYSTEM
Unified Flask + SocketIO server with REAL-TIME video streaming
Mobile camera → Desktop processing → Face recognition → Real-time feedback
"""

from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import cv2
import base64
import numpy as np
import time
import os
import logging
import uuid
import threading
from datetime import datetime

# Import services
try:
    from .services.face_service import FaceService
    from .services.employee_service import EmployeeService
    from .services.stream_service import StreamService
    from .models.models import DatabaseManager
    from .config import Config
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from services.face_service import FaceService
    from services.employee_service import EmployeeService
    from services.stream_service import StreamService
    from models.models import DatabaseManager
    from config import Config

# =====================================================
# APPLICATION SETUP
# =====================================================
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

app.config.from_object(Config)

# Initialize SocketIO với real-time optimization
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=10**8  # 100MB for video frames
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# INITIALIZE SERVICES
# =====================================================
try:
    # Database
    db_manager = DatabaseManager()
    
    # Core services
    face_service = FaceService()
    employee_service = EmployeeService(db_manager)
    stream_service = StreamService(socketio, face_service)
    
    logger.info("✅ All services initialized successfully")
except Exception as e:
    logger.error(f"❌ Service initialization failed: {e}")
    # Fallback to basic services
    face_service = None
    employee_service = None
    stream_service = None

# =====================================================
# GLOBAL STATE
# =====================================================
app_state = {
    'detection_active': True,
    'connected_clients': {},
    'desktop_monitors': set(),
    'stats': {
        'total_frames': 0,
        'total_detections': 0,
        'fps': 0,
        'uptime': time.time(),
        'session_start': datetime.now().isoformat()
    }
}

# =====================================================
# WEB ROUTES
# =====================================================

@app.route('/')
def desktop_interface():
    """Desktop monitoring interface"""
    return render_template('desktop.html')

@app.route('/mobile')
def mobile_interface():
    """Mobile camera interface"""
    return render_template('mobile.html')

@app.route('/health')
def health_check():
    """System health check"""
    health_data = {
        'status': 'healthy',
        'timestamp': time.time(),
        'uptime_seconds': time.time() - app_state['stats']['uptime'],
        'services': {
            'face_service': face_service is not None,
            'employee_service': employee_service is not None,
            'stream_service': stream_service is not None,
            'database': db_manager is not None if 'db_manager' in globals() else False
        },
        'stats': app_state['stats'],
        'connected_clients': len(app_state['connected_clients']),
        'detection_active': app_state['detection_active']
    }
    return jsonify(health_data)

# =====================================================
# API ROUTES - Employee Management
# =====================================================

@app.route('/api/employees', methods=['GET'])
def get_employees():
    """Get all employees with face data count"""
    try:
        if not employee_service:
            return jsonify({'success': False, 'error': 'Employee service not available'})
        
        employees = employee_service.get_all_employees()
        result = []
        
        for emp in employees:
            emp_data = {
                'id': emp.id,
                'name': emp.name,
                'employee_code': emp.employee_code,
                'department': emp.department,
                'position': emp.position,
                'email': emp.email,
                'phone': emp.phone,
                'created_at': emp.created_at.isoformat() if emp.created_at else None,
                'image_count': face_service.get_employee_image_count(emp.id) if face_service else 0
            }
            result.append(emp_data)
        
        return jsonify({'success': True, 'data': result, 'total': len(result)})
        
    except Exception as e:
        logger.error(f"❌ Error getting employees: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/employees', methods=['POST'])
def create_employee():
    """Create new employee"""
    try:
        if not employee_service:
            return jsonify({'success': False, 'error': 'Employee service not available'})
        
        data = request.get_json()
        
        # Validate required fields
        if not data.get('name') or not data.get('employee_code'):
            return jsonify({'success': False, 'error': 'Name and employee code are required'})
        
        employee = employee_service.create_employee(
            name=data['name'],
            employee_code=data['employee_code'],
            department=data.get('department'),
            position=data.get('position'),
            email=data.get('email'),
            phone=data.get('phone')
        )
        
        return jsonify({
            'success': True,
            'data': {
                'id': employee.id,
                'name': employee.name,
                'employee_code': employee.employee_code
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error creating employee: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/employees/<int:employee_id>/upload', methods=['POST'])
def upload_employee_image(employee_id):
    """Upload face image for employee"""
    try:
        if not face_service or not employee_service:
            return jsonify({'success': False, 'error': 'Services not available'})
        
        # Check if employee exists
        employee = employee_service.get_employee_by_id(employee_id)
        if not employee:
            return jsonify({'success': False, 'error': 'Employee not found'})
        
        # Check image upload
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Check current image count
        current_count = face_service.get_employee_image_count(employee_id)
        if current_count >= 10:
            return jsonify({'success': False, 'error': 'Maximum 10 images per employee'})
        
        # Save image and extract face vector
        vector_face = face_service.save_image_and_vector(file, employee_id)
        
        return jsonify({
            'success': True,
            'data': {
                'id': vector_face.id,
                'image_path': vector_face.image_path,
                'current_count': current_count + 1,
                'message': f'Image uploaded for {employee.name}'
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error uploading image: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/employees/<int:employee_id>', methods=['DELETE'])
def delete_employee(employee_id):
    """Delete employee and all face data"""
    try:
        if not employee_service:
            return jsonify({'success': False, 'error': 'Employee service not available'})
        
        employee = employee_service.get_employee_by_id(employee_id)
        if not employee:
            return jsonify({'success': False, 'error': 'Employee not found'})
        
        employee_name = employee.name
        
        # Delete face vectors first
        if face_service:
            face_service.delete_all_employee_vectors(employee_id)
        
        # Delete employee
        success = employee_service.delete_employee(employee_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Employee {employee_name} deleted successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to delete employee'})
        
    except Exception as e:
        logger.error(f"❌ Error deleting employee: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/vectors/<int:vector_id>', methods=['DELETE'])
def delete_face_vector(vector_id):
    """Delete specific face vector"""
    try:
        if not face_service:
            return jsonify({'success': False, 'error': 'Face service not available'})
        
        success = face_service.delete_face_vector(vector_id)
        
        if success:
            return jsonify({'success': True, 'message': 'Face vector deleted'})
        else:
            return jsonify({'success': False, 'error': 'Vector not found'})
        
    except Exception as e:
        logger.error(f"❌ Error deleting vector: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/employees/<int:employee_id>/vectors', methods=['GET'])
def get_employee_vectors(employee_id):
    """Get all face vectors for employee"""
    try:
        if not face_service:
            return jsonify({'success': False, 'error': 'Face service not available'})
        
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
        logger.error(f"❌ Error getting vectors: {e}")
        return jsonify({'success': False, 'error': str(e)})

# =====================================================
# API ROUTES - Streaming Control
# =====================================================

@app.route('/api/stream/stats', methods=['GET'])
def get_stream_stats():
    """Get real-time streaming statistics"""
    try:
        if stream_service:
            stats = stream_service.get_stats()
        else:
            stats = app_state['stats'].copy()
            stats['connected_clients'] = len(app_state['connected_clients'])
        
        stats['detection_active'] = app_state['detection_active']
        return jsonify({'success': True, 'data': stats})
        
    except Exception as e:
        logger.error(f"❌ Error getting stream stats: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stream/control', methods=['POST'])
def stream_control():
    """Control detection settings"""
    try:
        data = request.get_json()
        
        if 'detection_active' in data:
            app_state['detection_active'] = bool(data['detection_active'])
            
            # Broadcast to all clients
            socketio.emit('detection_status_changed', {
                'active': app_state['detection_active'],
                'timestamp': time.time()
            })
            
            return jsonify({
                'success': True,
                'detection_active': app_state['detection_active'],
                'message': f"Face detection {'enabled' if app_state['detection_active'] else 'disabled'}"
            })
        
        return jsonify({'success': False, 'error': 'Invalid control parameter'})
        
    except Exception as e:
        logger.error(f"❌ Error in stream control: {e}")
        return jsonify({'success': False, 'error': str(e)})

# =====================================================
# SOCKET.IO EVENTS - Connection Management
# =====================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = str(uuid.uuid4())
    session['client_id'] = client_id
    session['connected_at'] = time.time()
    
    logger.info(f'✅ Client connected: {client_id[:8]}...')
    
    emit('connection_established', {
        'client_id': client_id,
        'server_time': time.time(),
        'detection_active': app_state['detection_active'],
        'status': 'connected'
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = session.get('client_id', 'unknown')
    logger.info(f'❌ Client disconnected: {client_id[:8]}...')
    
    # Remove from tracking
    if client_id in app_state['connected_clients']:
        del app_state['connected_clients'][client_id]
    
    app_state['desktop_monitors'].discard(client_id)
    
    # Unregister from stream service
    if stream_service:
        stream_service.unregister_client(client_id)
    
    # Broadcast update
    socketio.emit('client_disconnected', {
        'client_id': client_id,
        'total_clients': len(app_state['connected_clients'])
    })

# =====================================================
# SOCKET.IO EVENTS - Mobile Client Management
# =====================================================

@socketio.on('register_mobile_client')
def handle_register_mobile(data):
    """Register mobile client for video streaming"""
    client_id = session.get('client_id', str(uuid.uuid4()))
    session['client_id'] = client_id
    session['client_type'] = 'mobile'
    
    client_info = {
        'type': 'mobile',
        'user_agent': data.get('user_agent', ''),
        'screen_size': data.get('screen_size', ''),
        'connected_at': time.time(),
        'frames_sent': 0,
        'last_frame_time': 0
    }
    
    app_state['connected_clients'][client_id] = client_info
    
    # Register with stream service
    if stream_service:
        stream_service.register_client(client_id, client_info)
    
    emit('mobile_registered', {
        'client_id': client_id,
        'detection_active': app_state['detection_active'],
        'server_ready': True
    })
    
    # Broadcast to desktop monitors
    socketio.emit('mobile_connected', {
        'client_id': client_id,
        'client_count': len(app_state['connected_clients']),
        'timestamp': time.time()
    }, room='desktop_monitors')
    
    logger.info(f'📱 Mobile client registered: {client_id[:8]}...')

@socketio.on('join_desktop_monitor')
def handle_join_desktop():
    """Desktop client joins monitoring room"""
    client_id = session.get('client_id')
    session['client_type'] = 'desktop'
    
    if client_id:
        join_room('desktop_monitors')
        app_state['desktop_monitors'].add(client_id)
        
        emit('desktop_monitor_joined', {
            'stats': app_state['stats'],
            'connected_clients': len(app_state['connected_clients']),
            'detection_active': app_state['detection_active'],
            'mobile_clients': [
                {
                    'id': cid,
                    'type': info['type'],
                    'connected_at': info['connected_at'],
                    'frames_sent': info.get('frames_sent', 0)
                }
                for cid, info in app_state['connected_clients'].items()
                if info.get('type') == 'mobile'
            ]
        })
        
        logger.info(f'🖥️ Desktop monitor joined: {client_id[:8]}...')

# =====================================================
# SOCKET.IO EVENTS - Real-time Video Processing
# =====================================================

@socketio.on('video_frame')
def handle_video_frame(data):
    """Process real-time video frame from mobile"""
    try:
        client_id = session.get('client_id', 'unknown')
        
        if client_id not in app_state['connected_clients']:
            emit('video_frame_error', {'error': 'Client not registered'})
            return
        
        frame_data = data.get('frame')
        if not frame_data:
            emit('video_frame_error', {'error': 'No frame data'})
            return
        
        # Update client stats
        client_info = app_state['connected_clients'][client_id]
        client_info['frames_sent'] += 1
        client_info['last_frame_time'] = time.time()
        
        # Update global stats
        app_state['stats']['total_frames'] += 1
        
        # Process frame with stream service
        if stream_service and face_service:
            result = stream_service.process_video_frame(client_id, frame_data, face_service)
        else:
            # Fallback basic processing
            result = _process_frame_basic(frame_data, data)
        
        # Send result back to mobile
        emit('detection_result', result)
        
        # Broadcast to desktop monitors
        socketio.emit('frame_processed', {
            'client_id': client_id,
            'detections': result.get('detections', []),
            'frame_count': app_state['stats']['total_frames'],
            'timestamp': time.time()
        }, room='desktop_monitors')
        
    except Exception as e:
        logger.error(f"❌ Video frame processing error: {e}")
        emit('video_frame_error', {'error': str(e)})

def _process_frame_basic(frame_data, metadata):
    """Basic frame processing fallback"""
    try:
        # Decode frame
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        detections = []
        
        if frame is not None and app_state['detection_active']:
            # Basic OpenCV face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                face_cascade = cv2.CascadeClassifier(cascade_path)
            except:
                # Fallback path
                face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                detections.append({
                    'type': 'unknown',
                    'name': 'Unknown Person',
                    'confidence': 0.75,
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'employee': None
                })
            
            app_state['stats']['total_detections'] += len(detections)
        
        return {
            'success': True,
            'timestamp': time.time(),
            'client_timestamp': metadata.get('timestamp'),
            'detections': detections,
            'detection_active': app_state['detection_active'],
            'frame_count': app_state['stats']['total_frames']
        }
        
    except Exception as e:
        logger.error(f"❌ Basic frame processing error: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }

# =====================================================
# SOCKET.IO EVENTS - System Control
# =====================================================

@socketio.on('toggle_detection')
def handle_toggle_detection(data):
    """Toggle face detection on/off"""
    app_state['detection_active'] = data.get('active', not app_state['detection_active'])
    
    socketio.emit('detection_status_changed', {
        'active': app_state['detection_active'],
        'timestamp': time.time(),
        'message': f"Face detection {'enabled' if app_state['detection_active'] else 'disabled'}"
    })
    
    logger.info(f"🔍 Face detection: {'ENABLED' if app_state['detection_active'] else 'DISABLED'}")

@socketio.on('get_system_stats')
def handle_get_stats():
    """Get current system statistics"""
    stats = app_state['stats'].copy()
    stats['connected_clients'] = len(app_state['connected_clients'])
    stats['desktop_monitors'] = len(app_state['desktop_monitors'])
    stats['detection_active'] = app_state['detection_active']
    stats['uptime_formatted'] = _format_uptime(time.time() - stats['uptime'])
    
    emit('system_stats_update', {
        'stats': stats,
        'timestamp': time.time()
    })

@socketio.on('ping')
def handle_ping(data):
    """Handle ping for latency measurement"""
    emit('pong', {
        'timestamp': data.get('timestamp', time.time()),
        'server_timestamp': time.time()
    })

# =====================================================
# SOCKET.IO EVENTS - Laptop Capture
# =====================================================

@socketio.on('capture_from_laptop')
def handle_laptop_capture(data):
    """Handle image capture from laptop camera"""
    try:
        employee_id = data.get('employee_id')
        image_data = data.get('image')
        
        if not employee_id or not image_data:
            emit('capture_result', {
                'success': False,
                'error': 'Employee ID and image data required'
            })
            return
        
        if not face_service or not employee_service:
            emit('capture_result', {
                'success': False,
                'error': 'Services not available'
            })
            return
        
        # Check if employee exists
        employee = employee_service.get_employee_by_id(employee_id)
        if not employee:
            emit('capture_result', {
                'success': False,
                'error': 'Employee not found'
            })
            return
        
        # Decode and process image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            emit('capture_result', {
                'success': False,
                'error': 'Could not decode image'
            })
            return
        
        # Save image and extract face vector
        # Convert frame to file-like object for face_service
        import io
        from PIL import Image
        
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Save to BytesIO
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        # Create a file-like object with required attributes
        class FileWrapper:
            def __init__(self, buffer, filename):
                self.buffer = buffer
                self.filename = filename
            
            def read(self):
                return self.buffer.read()
            
            def seek(self, pos):
                return self.buffer.seek(pos)
        
        file_wrapper = FileWrapper(img_buffer, f'laptop_capture_{employee_id}.jpg')
        vector_face = face_service.save_image_and_vector(file_wrapper, employee_id)
        
        emit('capture_result', {
            'success': True,
            'vector_id': vector_face.id,
            'employee_name': employee.name,
            'image_count': face_service.get_employee_image_count(employee_id),
            'message': f'Image captured and saved for {employee.name}'
        })
        
    except Exception as e:
        logger.error(f"❌ Laptop capture error: {e}")
        emit('capture_result', {
            'success': False,
            'error': str(e)
        })

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def _format_uptime(seconds):
    """Format uptime in human-readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# =====================================================
# ERROR HANDLERS
# =====================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@socketio.on_error_default
def default_error_handler(e):
    logger.error(f"SocketIO error: {e}")
    emit('error', {'message': 'Server error occurred'})

# =====================================================
# APPLICATION STARTUP
# =====================================================

def create_required_directories():
    """Create necessary directories"""
    directories = [
        app.config['UPLOAD_FOLDER'],
        'logs',
        os.path.dirname(app.config.get('DATABASE_URL', 'data/attendance.db'))
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")

def display_startup_info():
    """Display startup information"""
    print("\n" + "="*70)
    print("🚀 BHK TECH ATTENDANCE SYSTEM - REAL-TIME VIDEO STREAMING")
    print("="*70)
    print("✅ FEATURES ENABLED:")
    print("   • Real-time video streaming (Mobile → Desktop)")
    print("   • Face recognition with DeepFace")
    print("   • Employee management system")
    print("   • Live detection results with bounding boxes")
    print("   • Multi-client support")
    print("   • Desktop monitoring dashboard")
    print("="*70)
    print("💡 NETWORK SETUP:")
    print("   🪟 Windows: Get IP with 'ipconfig'")
    print("   🍎 Mac/Linux: Get IP with 'ifconfig'")
    print("   📱 Mobile URL: http://[YOUR_IP]:5000/mobile")
    print("="*70)
    print("🎯 SERVICES STATUS:")
    print(f"   • Database: {'✅ Connected' if 'db_manager' in globals() else '❌ Not available'}")
    print(f"   • Face Recognition: {'✅ Ready' if face_service else '❌ Basic mode'}")
    print(f"   • Employee Service: {'✅ Ready' if employee_service else '❌ Not available'}")
    print(f"   • Stream Service: {'✅ Ready' if stream_service else '❌ Basic mode'}")
    print("="*70)
    print(f"🖥️  Desktop Interface:  http://localhost:5000")
    print(f"📱 Mobile Interface:    http://[YOUR_IP]:5000/mobile")
    print(f"🔧 API Endpoints:       http://localhost:5000/api/")
    print(f"❤️  Health Check:       http://localhost:5000/health")
    print("="*70)
    print()

if __name__ == '__main__':
    try:
        # Setup
        create_required_directories()
        display_startup_info()
        
        # Start background tasks
        if stream_service:
            stream_service.start_background_tasks()
        
        # Start the server
        logger.info("🚀 Starting SocketIO server...")
        socketio.run(
            app,
            host='0.0.0.0',  # Allow external connections
            port=5000,
            debug=False,  # Set to False for production
            use_reloader=False,  # Prevent conflicts with SocketIO
            log_output=True
        )
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server startup error: {e}")
        raise
    finally:
        logger.info("👋 Server shutdown complete")