#!/usr/bin/env python3
"""
enhanced_app.py - BHK TECH ATTENDANCE SYSTEM WITH DEEPFACE INTEGRATION
Enhanced Flask + SocketIO server with commercial-grade face recognition
Real-time employee recognition using DeepFace and vector similarity matching
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

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# FLASK APP CONFIGURATION
# =====================================================
app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

# Enhanced configuration
app.config.update({
    'SECRET_KEY': 'BHK-Tech-Attendance-System-DeepFace-2024',
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
    logger=False,  # Reduce logging noise
    engineio_logger=False
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
    'mobile_clients': {},
    'desktop_clients': {},
    'current_frame': None,
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
# INITIALIZE ENHANCED SERVICES
# =====================================================

# Enhanced Face Service with DeepFace
try:
    from services.enhanced_face_service import EnhancedFaceService
    
    face_service = EnhancedFaceService(
        model_name='Facenet512',
        detector_backend='opencv',
        similarity_threshold=0.6,
        max_images_per_employee=10
    )
    
    logger.info("✅ Enhanced Face Service with DeepFace initialized")
    
except ImportError as e:
    logger.warning(f"⚠️ Could not import enhanced face service: {e}")
    logger.info("🔄 Using fallback face service")
    
    # Fallback to basic service
    class BasicFaceService:
        def __init__(self):
            self.face_cascade = None
            self.initialize_opencv()
        
        def initialize_opencv(self):
            try:
                import pkg_resources
                cascade_path = pkg_resources.resource_filename('cv2', 'data/haarcascade_frontalface_default.xml')
                if not os.path.exists(cascade_path):
                    # Fallback path
                    cascade_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    self.face_cascade = None
            except Exception as e:
                logger.error(f"OpenCV init failed: {e}")
        
        def recognize_faces_in_frame(self, frame):
            if self.face_cascade is None:
                return []
            
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                
                results = []
                for (x, y, w, h) in faces:
                    results.append({
                        'bbox': [int(x), int(y), int(w), int(h)],
                        'confidence': 0.8,
                        'employee': None,
                        'match_found': False,
                        'processing_time': 0.01,
                        'fallback_mode': True
                    })
                return results
            except Exception as e:
                logger.error(f"Fallback detection error: {e}")
                return []
        
        def save_employee_image_with_vector(self, file, employee_id):
            return {'success': False, 'error': 'Enhanced face service not available'}
        
        def get_employee_images(self, employee_id):
            return []
        
        def delete_employee_image(self, vector_id):
            return False
        
        def get_employee_image_count(self, employee_id):
            return 0
        
        def get_recognition_stats(self):
            return {
                'total_recognitions': 0,
                'successful_matches': 0,
                'success_rate': 0.0,
                'average_processing_time': 0.0,
                'deepface_available': False
            }
    
    face_service = BasicFaceService()

# Enhanced Employee Service
class EnhancedEmployeeService:
    def __init__(self):
        self.employees = [
            {
                'id': 1, 
                'name': 'Nguyễn Văn An', 
                'employee_code': 'EMP001',
                'department': 'IT Department',
                'position': 'Software Developer',
                'email': 'an.nguyen@bhktech.com',
                'phone': '0901234567',
                'status': 'active',
                'created_at': '2024-01-15'
            },
            {
                'id': 2, 
                'name': 'Trần Thị Bình', 
                'employee_code': 'EMP002',
                'department': 'HR Department',
                'position': 'HR Manager',
                'email': 'binh.tran@bhktech.com',
                'phone': '0901234568',
                'status': 'active',
                'created_at': '2024-01-15'
            },
            {
                'id': 3, 
                'name': 'Lê Văn Cường', 
                'employee_code': 'EMP003',
                'department': 'Finance Department',
                'position': 'Accountant',
                'email': 'cuong.le@bhktech.com',
                'phone': '0901234569',
                'status': 'active',
                'created_at': '2024-01-15'
            }
        ]
    
    def get_all_employees(self):
        return self.employees
    
    def create_employee(self, **kwargs):
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
        
        class MockEmployee:
            def __init__(self, emp_data):
                self.id = emp_data['id']
                self.__dict__.update(emp_data)
        
        return MockEmployee(new_employee)
    
    def delete_employee(self, employee_id):
        original_length = len(self.employees)
        self.employees = [emp for emp in self.employees if emp['id'] != employee_id]
        return len(self.employees) < original_length

employee_service = EnhancedEmployeeService()

# =====================================================
# REGISTER API ROUTES
# =====================================================
try:
    from api.enhanced_employee_routes import employee_bp, init_employee_routes
    
    # Initialize routes with services
    init_employee_routes(employee_service, face_service)
    
    # Register blueprint
    app.register_blueprint(employee_bp)
    
    logger.info("✅ Enhanced Employee API routes registered")
    
except ImportError as e:
    logger.warning(f"⚠️ Could not import enhanced employee routes: {e}")
    
    # Register basic API endpoints
    @app.route('/api/employees', methods=['GET'])
    def get_employees():
        try:
            employees = employee_service.get_all_employees()
            for emp in employees:
                emp['image_count'] = face_service.get_employee_image_count(emp['id'])
            return jsonify({'success': True, 'data': employees})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/employees', methods=['POST'])
    def create_employee():
        try:
            data = request.get_json()
            employee = employee_service.create_employee(**data)
            return jsonify({'success': True, 'data': {'id': employee.id}})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

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

def process_frame_with_recognition(frame):
    """Process frame with enhanced face recognition"""
    try:
        if frame is None:
            return None, []
        
        # Use enhanced face service for recognition
        detections = face_service.recognize_faces_in_frame(frame)
        
        # Draw detection boxes and labels
        processed_frame = frame.copy()
        
        for detection in detections:
            bbox = detection.get('bbox', [0, 0, 100, 100])
            x, y, w, h = bbox
            
            # Determine color based on recognition result
            if detection.get('match_found') and detection.get('employee'):
                # Known employee - green
                color = (0, 255, 0)
                employee = detection['employee']
                label = f"{employee['name']} ({detection['confidence']:.2f})"
            else:
                # Unknown person - red
                color = (0, 0, 255)
                label = f"Unknown ({detection['confidence']:.2f})"
            
            # Draw bounding box
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(processed_frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(processed_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
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

@app.route('/health')
def health_check():
    """System health check"""
    try:
        current_time = time.time()
        uptime = current_time - app_state['stats']['uptime']
        
        # Get face service stats
        recognition_stats = face_service.get_recognition_stats() if hasattr(face_service, 'get_recognition_stats') else {}
        
        health_data = {
            'status': 'healthy',
            'uptime_seconds': int(uptime),
            'system_stats': app_state['stats'],
            'face_recognition': recognition_stats,
            'services': {
                'face_service': face_service is not None,
                'employee_service': employee_service is not None,
                'deepface_available': getattr(face_service, 'DEEPFACE_AVAILABLE', False)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(health_data)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# =====================================================
# SOCKETIO EVENT HANDLERS
# =====================================================
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    from flask import session
    client_id = session.get('sid', 'unknown')
    logger.info(f"Client connected: {client_id}")
    
    emit('connection_status', {
        'connected': True,
        'client_id': client_id,
        'server_time': datetime.now().isoformat(),
        'detection_active': app_state['detection_active']
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    from flask import session
    client_id = session.get('sid', 'unknown')
    logger.info(f"Client disconnected: {client_id}")
    
    # Remove from tracking
    if client_id in app_state['mobile_clients']:
        del app_state['mobile_clients'][client_id]
        app_state['stats']['mobile_connections'] = len(app_state['mobile_clients'])
    
    if client_id in app_state['desktop_clients']:
        del app_state['desktop_clients'][client_id]
        app_state['stats']['desktop_connections'] = len(app_state['desktop_clients'])

@socketio.on('register_mobile_client')
def handle_mobile_registration(data):
    """Register mobile client for video streaming"""
    from flask import session
    client_id = session.get('sid', f'mobile_{int(time.time())}')
    
    app_state['mobile_clients'][client_id] = {
        'id': client_id,
        'registered_at': datetime.now().isoformat(),
        'device_info': data.get('device_info', {}),
        'last_frame_time': None
    }
    
    app_state['stats']['mobile_connections'] = len(app_state['mobile_clients'])
    
    logger.info(f"📱 Mobile client registered: {client_id}")
    
    join_room('mobile_clients')
    
    emit('mobile_registration_confirmed', {
        'client_id': client_id,
        'status': 'registered',
        'detection_active': app_state['detection_active'],
        'timestamp': datetime.now().isoformat()
    })
    
    # Notify desktop clients
    socketio.emit('mobile_client_connected', {
        'client_id': client_id,
        'device_info': data.get('device_info', {}),
        'timestamp': datetime.now().isoformat()
    }, to='desktop_monitors')

@socketio.on('register_desktop_monitor')
def handle_desktop_registration(data):
    """Register desktop client for monitoring"""
    from flask import session
    client_id = session.get('sid', f'desktop_{int(time.time())}')
    
    app_state['desktop_clients'][client_id] = {
        'id': client_id,
        'registered_at': datetime.now().isoformat(),
        'monitor_settings': data.get('settings', {}),
        'last_activity': datetime.now().isoformat()
    }
    
    app_state['stats']['desktop_connections'] = len(app_state['desktop_clients'])
    
    logger.info(f"🖥️ Desktop monitor registered: {client_id}")
    
    join_room('desktop_monitors')
    
    emit('desktop_registration_confirmed', {
        'client_id': client_id,
        'status': 'registered',
        'detection_active': app_state['detection_active'],
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

@socketio.on('video_frame')
def handle_video_frame(data):
    """
    Process video frame from mobile with enhanced DeepFace recognition
    """
    from flask import session
    client_id = session.get('sid', f'frame_{int(time.time())}')
    
    try:
        # Validate mobile client
        if client_id not in app_state['mobile_clients']:
            logger.warning(f"Unregistered mobile client sent frame: {client_id}")
            emit('detection_result', {'success': False, 'error': 'Client not registered'})
            return
        
        frame_data = data.get('frame')
        if not frame_data:
            emit('detection_result', {'success': False, 'error': 'No frame data'})
            return
        
        # Update mobile client activity
        app_state['mobile_clients'][client_id]['last_frame_time'] = datetime.now().isoformat()
        
        # Decode frame
        frame = decode_base64_frame(frame_data)
        if frame is None:
            emit('detection_result', {'success': False, 'error': 'Failed to decode frame'})
            return
        
        # Process frame with enhanced recognition
        processed_frame_data = frame_data
        recognition_results = []
        
        if app_state['detection_active']:
            try:
                # Use enhanced face service for recognition
                processed_frame, recognition_results = process_frame_with_recognition(frame)
                
                if processed_frame is not None:
                    processed_frame_data = encode_frame_to_base64(processed_frame)
                
                # Update statistics
                app_state['stats']['total_detections'] += len(recognition_results)
                
                # Count successful recognitions
                successful_recognitions = sum(1 for r in recognition_results if r.get('match_found'))
                app_state['stats']['total_recognitions'] += successful_recognitions
                
                logger.debug(f"🎯 Processed frame: {len(recognition_results)} faces, {successful_recognitions} recognized")
                
            except Exception as recognition_error:
                logger.error(f"❌ Recognition error: {recognition_error}")
                recognition_results = []
        
        # Update global frame state
        with frame_lock:
            app_state['current_frame'] = processed_frame_data
            app_state['frame_timestamp'] = datetime.now().isoformat()
            app_state['stats']['total_frames'] += 1
            app_state['stats']['last_frame_time'] = app_state['frame_timestamp']
        
        # Broadcast to desktop monitors
        socketio.emit('video_frame_update', {
            'frame': processed_frame_data,
            'timestamp': app_state['frame_timestamp'],
            'source': 'mobile',
            'client_id': client_id,
            'detections': recognition_results,
            'detection_active': app_state['detection_active'],
            'processing_stats': {
                'faces_detected': len(recognition_results),
                'employees_recognized': sum(1 for r in recognition_results if r.get('match_found')),
                'total_frames': app_state['stats']['total_frames']
            }
        }, to='desktop_monitors')
        
        # Send result back to mobile with employee details
        mobile_result = {
            'success': True,
            'faces': [],
            'timestamp': datetime.now().isoformat(),
            'frame_count': app_state['stats']['total_frames'],
            'detection_active': app_state['detection_active']
        }
        
        # Format recognition results for mobile display
        for result in recognition_results:
            face_data = {
                'x': result['bbox'][0],
                'y': result['bbox'][1],
                'width': result['bbox'][2],
                'height': result['bbox'][3],
                'confidence': result.get('confidence', 0.0),
                'match_found': result.get('match_found', False)
            }
            
            # Add employee information if recognized
            if result.get('employee'):
                face_data['employee'] = {
                    'id': result['employee']['id'],
                    'name': result['employee']['name'],
                    'employee_code': result['employee']['employee_code'],
                    'department': result['employee']['department']
                }
            
            mobile_result['faces'].append(face_data)
        
        emit('detection_result', mobile_result)
        
        logger.debug(f"📱 Sent result to mobile: {len(mobile_result['faces'])} faces")
        
    except Exception as e:
        logger.error(f"❌ Error in video_frame handler: {e}")
        emit('detection_result', {'success': False, 'error': str(e)})

@socketio.on('toggle_detection')
def handle_toggle_detection(data):
    """Toggle face detection on/off"""
    from flask import session
    client_id = session.get('sid', f'toggle_{int(time.time())}')
    
    # Toggle detection state
    app_state['detection_active'] = not app_state['detection_active']
    
    logger.info(f"🔍 Detection toggled by {client_id}: {app_state['detection_active']}")
    
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

@socketio.on('set_detection_active')
def handle_set_detection_active(data):
    """Set detection state explicitly"""
    from flask import session
    client_id = session.get('sid', f'set_{int(time.time())}')
    new_state = data.get('active', False)
    
    app_state['detection_active'] = new_state
    
    logger.info(f"🔍 Detection set to {new_state} by {client_id}")
    
    # Broadcast state change
    socketio.emit('detection_state_changed', {
        'detection_active': app_state['detection_active'],
        'timestamp': datetime.now().isoformat(),
        'changed_by': client_id
    })

@socketio.on('get_system_status')
def handle_get_status(data):
    """Get current system status"""
    try:
        current_time = time.time()
        uptime = current_time - app_state['stats']['uptime']
        
        # Get recognition stats from face service
        recognition_stats = {}
        if hasattr(face_service, 'get_recognition_stats'):
            recognition_stats = face_service.get_recognition_stats()
        
        status = {
            'detection_active': app_state['detection_active'],
            'mobile_clients': len(app_state['mobile_clients']),
            'desktop_clients': len(app_state['desktop_clients']),
            'has_current_frame': app_state['current_frame'] is not None,
            'system_stats': {
                'total_frames': app_state['stats']['total_frames'],
                'total_detections': app_state['stats']['total_detections'],
                'total_recognitions': app_state['stats']['total_recognitions'],
                'uptime_seconds': int(uptime),
                'last_frame_time': app_state['stats']['last_frame_time']
            },
            'recognition_stats': recognition_stats,
            'services': {
                'face_service_type': 'enhanced' if hasattr(face_service, 'model_name') else 'basic',
                'deepface_available': getattr(face_service, 'DEEPFACE_AVAILABLE', False),
                'model_name': getattr(face_service, 'model_name', 'N/A'),
                'similarity_threshold': getattr(face_service, 'similarity_threshold', 0.0)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        emit('system_status_update', status)
        
    except Exception as e:
        logger.error(f"❌ Error getting system status: {e}")
        emit('error', {'message': f'Status error: {str(e)}'})

@socketio.on('get_recognition_stats')
def handle_get_recognition_stats(data):
    """Get detailed recognition statistics"""
    try:
        if hasattr(face_service, 'get_recognition_stats'):
            stats = face_service.get_recognition_stats()
            
            # Add system-level stats
            stats['system_stats'] = {
                'total_frames_processed': app_state['stats']['total_frames'],
                'total_faces_detected': app_state['stats']['total_detections'],
                'total_employees_recognized': app_state['stats']['total_recognitions'],
                'current_fps': app_state['stats'].get('fps', 0)
            }
            
            emit('recognition_stats_update', {
                'success': True,
                'data': stats,
                'timestamp': datetime.now().isoformat()
            })
        else:
            emit('recognition_stats_update', {
                'success': False,
                'error': 'Enhanced recognition stats not available'
            })
            
    except Exception as e:
        logger.error(f"❌ Error getting recognition stats: {e}")
        emit('error', {'message': f'Recognition stats error: {str(e)}'})

@socketio.on('ping')
def handle_ping(data):
    """Handle ping for connection testing"""
    emit('pong', {
        'timestamp': datetime.now().isoformat(),
        'server_time': time.time(),
        'client_data': data
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
    return jsonify({'error': 'Internal server error', 'timestamp': datetime.now().isoformat()}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

# =====================================================
# APPLICATION INITIALIZATION
# =====================================================
def initialize_application():
    """Initialize application components"""
    try:
        # Create upload directory
        upload_dir = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_dir, exist_ok=True)
        logger.info(f"📁 Upload directory ready: {upload_dir}")
        
        # Test face service
        logger.info("🧪 Testing face service...")
        if hasattr(face_service, 'get_recognition_stats'):
            stats = face_service.get_recognition_stats()
            logger.info(f"   - DeepFace: {'Available' if stats.get('deepface_available') else 'Not Available'}")
            logger.info(f"   - Model: {stats.get('model_name', 'Basic OpenCV')}")
        
        # Log employee count
        employees = employee_service.get_all_employees()
        logger.info(f"👥 Employee database: {len(employees)} employees loaded")
        
        # Update app config with current time
        app.config['CURRENT_TIME'] = datetime.now().isoformat()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        return False

def print_startup_banner():
    """Print startup banner with system info"""
    print("🚀 " + "="*60)
    print("    BHK TECH ATTENDANCE SYSTEM - DEEPFACE EDITION")
    print("="*64)
    print("✅ FEATURES:")
    print("   • Real-time mobile-to-desktop video streaming")
    print("   • Commercial-grade face recognition with DeepFace")
    print("   • Employee database with vector similarity matching")
    print("   • Multi-client support with SocketIO")
    print("   • Advanced face detection and recognition analytics")
    print("")
    print("🔗 INTERFACES:")
    print(f"   📱 Mobile Interface:  http://localhost:5000/mobile")
    print(f"   🖥️  Desktop Monitor:   http://localhost:5000/")
    print(f"   🔧 API Endpoints:     http://localhost:5000/api/")
    print(f"   ❤️  Health Check:     http://localhost:5000/health")
    print("")
    print("🎯 DEEPFACE INTEGRATION:")
    
    if hasattr(face_service, 'model_name'):
        print(f"   • Model: {getattr(face_service, 'model_name', 'N/A')}")
        print(f"   • Detector: {getattr(face_service, 'detector_backend', 'N/A')}")
        print(f"   • Threshold: {getattr(face_service, 'similarity_threshold', 'N/A')}")
        print(f"   • Status: Enhanced Recognition Active")
    else:
        print("   • Status: Basic Detection Mode (Install DeepFace for full features)")
    
    print("")
    print("💡 USAGE:")
    print("   1. Open desktop interface for monitoring")
    print("   2. Add employees and upload their photos")
    print("   3. Access mobile interface on phone/tablet")
    print("   4. Start camera streaming for real-time recognition")
    print("")
    print("🎉 READY FOR COMMERCIAL USE!")
    print("="*64)

# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == '__main__':
    # Print startup banner
    print_startup_banner()
    
    # Initialize application
    if not initialize_application():
        print("❌ Failed to initialize application")
        sys.exit(1)
    
    print("✅ Application initialized successfully")
    print("🚀 Starting server...")
    print("")
    
    try:
        # Start the server
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for production stability
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        print("👋 BHK Tech Attendance System shutdown complete")