# backend/services/stream_service.py
import time
import threading
from collections import defaultdict
import cv2
import numpy as np
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamService:
    """
    Real-time video streaming service
    Handles mobile video frames and coordinates with face recognition
    """
    
    def __init__(self, socketio):
        self.socketio = socketio
        self.clients = {}
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'start_time': time.time(),
            'last_fps_update': time.time(),
            'fps': 0,
            'frame_count_for_fps': 0
        }
        self.settings = {
            'max_fps': 15,
            'max_clients': 10,
            'frame_quality': 0.7,
            'detection_enabled': True
        }
        
        logger.info("StreamService initialized")
    
    def register_client(self, client_id, client_info):
        """Register a new mobile client"""
        self.clients[client_id] = {
            'info': client_info,
            'connected_at': time.time(),
            'frames_received': 0,
            'last_frame_time': 0,
            'fps': 0
        }
        
        logger.info(f"📱 Client registered: {client_id[:8]}...")
        return len(self.clients)
    
    def unregister_client(self, client_id):
        """Unregister a mobile client"""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"📱 Client unregistered: {client_id[:8]}...")
        
        return len(self.clients)
    
    def process_video_frame(self, client_id, frame_data, face_service):
        """
        Process incoming video frame from mobile client
        Returns detection results for real-time display
        """
        try:
            if client_id not in self.clients:
                return {'success': False, 'error': 'Client not registered'}
            
            # Update client stats
            client = self.clients[client_id]
            client['frames_received'] += 1
            client['last_frame_time'] = time.time()
            
            # Decode frame
            frame = self._decode_frame(frame_data)
            if frame is None:
                return {'success': False, 'error': 'Could not decode frame'}
            
            # Update global stats
            self.stats['total_frames'] += 1
            self.stats['frame_count_for_fps'] += 1
            
            # Face detection (if enabled)
            detections = []
            if self.settings['detection_enabled']:
                detections = face_service.recognize_face(frame)
                self.stats['total_detections'] += len(detections)
            
            # Update FPS calculation
            self.update_fps_stats()
            
            # Prepare response
            result = {
                'success': True,
                'timestamp': time.time(),
                'detections': detections,
                'client_stats': {
                    'frames_received': client['frames_received'],
                    'fps': self._calculate_client_fps(client_id)
                },
                'detection_enabled': self.settings['detection_enabled']
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Frame processing error for {client_id[:8]}...: {e}")
            return {'success': False, 'error': str(e)}
    
    def _decode_frame(self, frame_data):
        """Decode base64 video frame"""
        try:
            # Remove data URL prefix if present
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            # Decode base64
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return frame
            
        except Exception as e:
            logger.error(f"Frame decode error: {e}")
            return None
    
    def _calculate_client_fps(self, client_id):
        """Calculate FPS for specific client"""
        if client_id not in self.clients:
            return 0
        
        client = self.clients[client_id]
        time_diff = time.time() - client['connected_at']
        
        if time_diff > 0:
            return round(client['frames_received'] / time_diff, 1)
        
        return 0
    
    def update_fps_stats(self):
        """Update global FPS statistics"""
        current_time = time.time()
        time_diff = current_time - self.stats['last_fps_update']
        
        # Update FPS every second
        if time_diff >= 1.0:
            self.stats['fps'] = round(self.stats['frame_count_for_fps'] / time_diff, 1)
            self.stats['last_fps_update'] = current_time
            self.stats['frame_count_for_fps'] = 0
    
    def get_stats(self):
        """Get current streaming statistics"""
        uptime = time.time() - self.stats['start_time']
        
        return {
            'total_frames': self.stats['total_frames'],
            'total_detections': self.stats['total_detections'],
            'fps': self.stats['fps'],
            'connected_clients': len(self.clients),
            'uptime_seconds': uptime,
            'uptime_formatted': self._format_uptime(uptime),
            'clients': {
                client_id: {
                    'frames_received': client['frames_received'],
                    'fps': self._calculate_client_fps(client_id),
                    'connected_duration': time.time() - client['connected_at']
                }
                for client_id, client in self.clients.items()
            }
        }
    
    def _format_uptime(self, seconds):
        """Format uptime in human readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def set_detection_enabled(self, enabled):
        """Enable/disable face detection"""
        self.settings['detection_enabled'] = enabled
        logger.info(f"🔍 Face detection: {'ENABLED' if enabled else 'DISABLED'}")
        
        # Broadcast to all clients
        self.socketio.emit('detection_status_changed', {
            'active': enabled,
            'timestamp': time.time()
        })
    
    def is_detection_enabled(self):
        """Check if face detection is enabled"""
        return self.settings['detection_enabled']
    
    def update_settings(self, new_settings):
        """Update streaming settings"""
        self.settings.update(new_settings)
        logger.info(f"Settings updated: {new_settings}")
    
    def get_client_count(self):
        """Get number of connected clients"""
        return len(self.clients)
    
    def cleanup_inactive_clients(self, timeout_seconds=30):
        """Remove clients that haven't sent frames recently"""
        current_time = time.time()
        inactive_clients = []
        
        for client_id, client in self.clients.items():
            if current_time - client['last_frame_time'] > timeout_seconds:
                inactive_clients.append(client_id)
        
        for client_id in inactive_clients:
            self.unregister_client(client_id)
            logger.info(f"📱 Removed inactive client: {client_id[:8]}...")
        
        return len(inactive_clients)
    
    def broadcast_stats(self):
        """Broadcast current stats to all connected desktop clients"""
        stats = self.get_stats()
        self.socketio.emit('stats_update', {
            'stats': stats,
            'timestamp': time.time()
        }, room='desktop_monitors')
    
    def get_network_info(self):
        """Get network information for mobile connection"""
        import socket
        
        try:
            # Get local IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            return {
                'local_ip': local_ip,
                'port': 5000,
                'mobile_url': f'http://{local_ip}:5000/mobile'
            }
        except Exception as e:
            logger.error(f"Could not get network info: {e}")
            return {
                'local_ip': 'localhost',
                'port': 5000,
                'mobile_url': 'http://localhost:5000/mobile'
            }
    
    def validate_frame_rate(self, client_id):
        """Check if client is sending frames too fast"""
        if client_id not in self.clients:
            return True
        
        client = self.clients[client_id]
        current_time = time.time()
        
        # Calculate time since last frame
        if client['last_frame_time'] > 0:
            time_diff = current_time - client['last_frame_time']
            min_interval = 1.0 / self.settings['max_fps']
            
            if time_diff < min_interval:
                return False  # Too fast
        
        return True
    
    def get_frame_processing_queue_size(self):
        """Get current frame processing queue size (for monitoring)"""
        # This would be implemented if using a queue-based processing system
        return 0
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'start_time': time.time(),
            'last_fps_update': time.time(),
            'fps': 0,
            'frame_count_for_fps': 0
        }
        
        # Reset client stats
        for client in self.clients.values():
            client['frames_received'] = 0
            client['connected_at'] = time.time()
        
        logger.info("📊 Statistics reset")
    
    def export_session_data(self):
        """Export session data for analysis"""
        session_data = {
            'session_info': {
                'start_time': self.stats['start_time'],
                'end_time': time.time(),
                'duration': time.time() - self.stats['start_time']
            },
            'stats': self.get_stats(),
            'settings': self.settings.copy(),
            'clients_summary': {
                'total_clients': len(self.clients),
                'client_details': [
                    {
                        'client_id': client_id[:8] + '...',
                        'frames_received': client['frames_received'],
                        'connection_duration': time.time() - client['connected_at'],
                        'avg_fps': self._calculate_client_fps(client_id)
                    }
                    for client_id, client in self.clients.items()
                ]
            }
        }
        
        return session_data


class FpsCalculator:
    """Helper class for FPS calculation"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
        self.last_fps = 0
    
    def update(self):
        """Update FPS calculation with new frame"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only recent frames
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        # Calculate FPS
        if len(self.frame_times) >= 2:
            time_span = self.frame_times[-1] - self.frame_times[0]
            if time_span > 0:
                self.last_fps = (len(self.frame_times) - 1) / time_span
        
        return self.last_fps
    
    def get_fps(self):
        """Get current FPS"""
        return round(self.last_fps, 1)


class PerformanceMonitor:
    """Monitor system performance during streaming"""
    
    def __init__(self):
        self.metrics = {
            'frame_processing_times': [],
            'detection_times': [],
            'average_processing_time': 0,
            'peak_processing_time': 0,
            'total_processed_frames': 0
        }
        self.lock = threading.Lock()
    
    def record_frame_processing(self, processing_time):
        """Record frame processing time"""
        with self.lock:
            self.metrics['frame_processing_times'].append(processing_time)
            self.metrics['total_processed_frames'] += 1
            
            # Keep only last 100 measurements
            if len(self.metrics['frame_processing_times']) > 100:
                self.metrics['frame_processing_times'].pop(0)
            
            # Update statistics
            times = self.metrics['frame_processing_times']
            self.metrics['average_processing_time'] = sum(times) / len(times)
            self.metrics['peak_processing_time'] = max(times)
    
    def record_detection_time(self, detection_time):
        """Record face detection time"""
        with self.lock:
            self.metrics['detection_times'].append(detection_time)
            
            # Keep only last 100 measurements
            if len(self.metrics['detection_times']) > 100:
                self.metrics['detection_times'].pop(0)
    
    def get_performance_stats(self):
        """Get current performance statistics"""
        with self.lock:
            return {
                'average_processing_time_ms': round(self.metrics['average_processing_time'] * 1000, 2),
                'peak_processing_time_ms': round(self.metrics['peak_processing_time'] * 1000, 2),
                'total_processed_frames': self.metrics['total_processed_frames'],
                'average_detection_time_ms': round(
                    sum(self.metrics['detection_times']) / len(self.metrics['detection_times']) * 1000, 2
                ) if self.metrics['detection_times'] else 0
            }
    
    def reset(self):
        """Reset all performance metrics"""
        with self.lock:
            self.metrics = {
                'frame_processing_times': [],
                'detection_times': [],
                'average_processing_time': 0,
                'peak_processing_time': 0,
                'total_processed_frames': 0
            }


def initialize_stream_service(socketio):
    """Initialize the stream service with proper configuration"""
    service = StreamService(socketio)
    
    # Start periodic cleanup task
    def cleanup_task():
        while True:
            try:
                removed_count = service.cleanup_inactive_clients()
                if removed_count > 0:
                    logger.info(f"🧹 Cleaned up {removed_count} inactive clients")
                
                # Broadcast stats every 5 seconds
                service.broadcast_stats()
                
                time.sleep(5)
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                time.sleep(5)
    
    # Start background thread
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()
    
    logger.info("🚀 StreamService initialized with background tasks")
    return service