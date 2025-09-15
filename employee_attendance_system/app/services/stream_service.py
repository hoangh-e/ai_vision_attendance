#!/usr/bin/env python3
"""
stream_service.py - REAL-TIME VIDEO STREAMING SERVICE
High-performance video streaming with face recognition
Multi-client support and real-time detection results
"""

import cv2
import numpy as np
import base64
import time
import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid
import queue
import asyncio
from collections import defaultdict, deque

# Database imports
from ..models.models import AttendanceLog, StreamSession, DatabaseManager
from ..database.database import db_session_scope
from ..config import Config

logger = logging.getLogger(__name__)


class StreamService:
    """Real-time video streaming service with face recognition"""
    
    def __init__(self, socketio, face_service=None):
        self.socketio = socketio
        self.face_service = face_service
        self.config = Config()
        
        # Client management
        self.clients = {}  # client_id -> client_info
        self.client_queues = {}  # client_id -> frame_queue
        self.client_stats = defaultdict(lambda: {
            'frames_processed': 0,
            'detections_found': 0,
            'avg_processing_time': 0.0,
            'last_frame_time': 0,
            'fps': 0.0,
            'frame_times': deque(maxlen=30)  # For FPS calculation
        })
        
        # Processing configuration
        self.max_clients = self.config.STREAM_MAX_CLIENTS
        self.frame_timeout = self.config.STREAM_FRAME_TIMEOUT
        self.max_fps = self.config.STREAM_MAX_FPS
        self.frame_interval = 1.0 / self.max_fps
        
        # Background processing
        self.processing_thread = None
        self.cleanup_thread = None
        self.running = False
        
        # Performance tracking
        self.global_stats = {
            'total_frames_processed': 0,
            'total_detections': 0,
            'total_clients_served': 0,
            'service_start_time': time.time(),
            'last_cleanup': time.time(),
            'peak_concurrent_clients': 0
        }
        
        logger.info("✅ StreamService initialized")
    
    def register_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """Register new streaming client"""
        try:
            if len(self.clients) >= self.max_clients:
                logger.warning(f"Max clients ({self.max_clients}) reached, rejecting {client_id}")
                return False
            
            # Initialize client
            self.clients[client_id] = {
                **client_info,
                'registered_at': time.time(),
                'last_activity': time.time(),
                'status': 'active'
            }
            
            # Initialize frame queue for client
            self.client_queues[client_id] = queue.Queue(maxsize=5)  # Buffer up to 5 frames
            
            # Update stats
            self.global_stats['total_clients_served'] += 1
            current_clients = len(self.clients)
            if current_clients > self.global_stats['peak_concurrent_clients']:
                self.global_stats['peak_concurrent_clients'] = current_clients
            
            # Save session to database
            self._save_stream_session(client_id, client_info)
            
            logger.info(f"✅ Registered client {client_id[:8]}... ({current_clients}/{self.max_clients})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error registering client {client_id}: {e}")
            return False
    
    def unregister_client(self, client_id: str) -> bool:
        """Unregister streaming client"""
        try:
            if client_id not in self.clients:
                return False
            
            # Clean up client data
            client_info = self.clients.pop(client_id, {})
            
            # Clean up frame queue
            if client_id in self.client_queues:
                del self.client_queues[client_id]
            
            # Clean up stats
            if client_id in self.client_stats:
                del self.client_stats[client_id]
            
            # Update session in database
            self._end_stream_session(client_id)
            
            logger.info(f"✅ Unregistered client {client_id[:8]}... ({len(self.clients)}/{self.max_clients})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error unregistering client {client_id}: {e}")
            return False
    
    def process_video_frame(self, client_id: str, frame_data: str, face_service=None) -> Dict[str, Any]:
        """Process single video frame with face recognition"""
        start_time = time.time()
        
        try:
            # Validate client
            if client_id not in self.clients:
                return {
                    'success': False,
                    'error': 'Client not registered',
                    'timestamp': time.time()
                }
            
            # Update client activity
            self.clients[client_id]['last_activity'] = time.time()
            
            # Check frame rate limiting
            if not self._should_process_frame(client_id):
                return {
                    'success': True,
                    'skipped': True,
                    'reason': 'Frame rate limited',
                    'timestamp': time.time()
                }
            
            # Decode frame
            frame = self._decode_frame(frame_data)
            if frame is None:
                return {
                    'success': False,
                    'error': 'Could not decode frame',
                    'timestamp': time.time()
                }
            
            # Process with face recognition
            detections = []
            if face_service:
                detections = face_service.detect_and_recognize_faces(frame)
            else:
                # Fallback to basic detection
                detections = self._basic_face_detection(frame)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_client_stats(client_id, processing_time, len(detections))
            
            # Save attendance logs for recognized faces
            self._save_attendance_logs(client_id, detections, processing_time)
            
            # Prepare response
            result = {
                'success': True,
                'timestamp': time.time(),
                'processing_time': processing_time,
                'detections': self._format_detections(detections),
                'client_stats': self._get_client_stats(client_id),
                'frame_count': self.client_stats[client_id]['frames_processed']
            }
            
            # Broadcast to desktop monitors
            self._broadcast_detection_result(client_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing frame for {client_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time(),
                'processing_time': time.time() - start_time
            }
    
    def _should_process_frame(self, client_id: str) -> bool:
        """Check if frame should be processed based on rate limiting"""
        stats = self.client_stats[client_id]
        current_time = time.time()
        
        if stats['last_frame_time'] == 0:
            stats['last_frame_time'] = current_time
            return True
        
        time_since_last = current_time - stats['last_frame_time']
        if time_since_last >= self.frame_interval:
            stats['last_frame_time'] = current_time
            return True
        
        return False
    
    def _decode_frame(self, frame_data: str) -> Optional[np.ndarray]:
        """Decode base64 frame data to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if ',' in frame_data:
                frame_data = frame_data.split(',', 1)[1]
            
            # Decode base64
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            
            # Decode image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error("Failed to decode frame data")
                return None
            
            # Resize if too large (performance optimization)
            h, w = frame.shape[:2]
            max_size = self.config.MOBILE_MAX_RESOLUTION
            
            if w > max_size[0] or h > max_size[1]:
                scale = min(max_size[0]/w, max_size[1]/h)
                new_w, new_h = int(w*scale), int(h*scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            return frame
            
        except Exception as e:
            logger.error(f"❌ Frame decode error: {e}")
            return None
    
    def _basic_face_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Fallback basic face detection using OpenCV"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            
            detections = []
            for (x, y, w, h) in faces:
                detections.append({
                    'type': 'unknown',
                    'employee': None,
                    'name': 'Unknown Person',
                    'confidence': 0.75,
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'embedding': None,
                    'detection_method': 'opencv_basic'
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Basic face detection error: {e}")
            return []
    
    def _format_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format detections for client response"""
        formatted = []
        
        for detection in detections:
            formatted_detection = {
                'type': detection.get('type', 'unknown'),
                'name': detection.get('name', 'Unknown'),
                'confidence': round(detection.get('confidence', 0.0), 3),
                'bbox': detection.get('bbox', [0, 0, 0, 0]),
                'detection_method': detection.get('detection_method', 'unknown')
            }
            
            # Add employee info if recognized
            if detection.get('employee'):
                emp = detection['employee']
                formatted_detection['employee'] = {
                    'id': emp.get('employee_id'),
                    'name': emp.get('name'),
                    'code': emp.get('employee_code'),
                    'department': emp.get('department')
                }
            
            formatted.append(formatted_detection)
        
        return formatted
    
    def _update_client_stats(self, client_id: str, processing_time: float, detection_count: int):
        """Update client performance statistics"""
        stats = self.client_stats[client_id]
        current_time = time.time()
        
        # Update counters
        stats['frames_processed'] += 1
        stats['detections_found'] += detection_count
        
        # Update processing time average
        if stats['avg_processing_time'] == 0:
            stats['avg_processing_time'] = processing_time
        else:
            stats['avg_processing_time'] = (
                stats['avg_processing_time'] * 0.9 + processing_time * 0.1
            )
        
        # Update FPS calculation
        stats['frame_times'].append(current_time)
        if len(stats['frame_times']) >= 2:
            time_span = stats['frame_times'][-1] - stats['frame_times'][0]
            if time_span > 0:
                stats['fps'] = (len(stats['frame_times']) - 1) / time_span
        
        # Update global stats
        self.global_stats['total_frames_processed'] += 1
        self.global_stats['total_detections'] += detection_count
    
    def _get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Get formatted client statistics"""
        if client_id not in self.client_stats:
            return {}
        
        stats = self.client_stats[client_id]
        return {
            'frames_processed': stats['frames_processed'],
            'detections_found': stats['detections_found'],
            'avg_processing_time': round(stats['avg_processing_time'], 3),
            'current_fps': round(stats['fps'], 1),
            'detection_rate': (
                round(stats['detections_found'] / stats['frames_processed'], 3)
                if stats['frames_processed'] > 0 else 0.0
            )
        }
    
    def _save_attendance_logs(self, client_id: str, detections: List[Dict[str, Any]], processing_time: float):
        """Save attendance logs for recognized employees"""
        try:
            recognized_detections = [
                d for d in detections 
                if d.get('type') == 'recognized' and d.get('employee')
            ]
            
            if not recognized_detections:
                return
            
            with db_session_scope() as session:
                for detection in recognized_detections:
                    employee = detection['employee']
                    
                    # Create attendance log
                    log = AttendanceLog(
                        employee_id=employee.get('employee_id'),
                        confidence_score=detection.get('confidence', 0.0),
                        detection_method='realtime',
                        client_id=client_id,
                        processing_time=processing_time
                    )
                    
                    # Set bounding box if available
                    if detection.get('bbox'):
                        log.bbox_coordinates = detection['bbox']
                    
                    session.add(log)
                
                session.commit()
                logger.debug(f"✅ Saved {len(recognized_detections)} attendance logs")
                
        except Exception as e:
            logger.error(f"❌ Error saving attendance logs: {e}")
    
    def _save_stream_session(self, client_id: str, client_info: Dict[str, Any]):
        """Save stream session to database"""
        try:
            with db_session_scope() as session:
                stream_session = StreamSession(
                    client_id=client_id,
                    client_type=client_info.get('type', 'unknown'),
                    user_agent=client_info.get('user_agent', ''),
                    screen_size=client_info.get('screen_size', ''),
                    is_active=True
                )
                
                session.add(stream_session)
                session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error saving stream session: {e}")
    
    def _end_stream_session(self, client_id: str):
        """End stream session in database"""
        try:
            with db_session_scope() as session:
                stream_session = session.query(StreamSession).filter(
                    StreamSession.client_id == client_id,
                    StreamSession.is_active == True
                ).first()
                
                if stream_session:
                    stream_session.end_time = datetime.utcnow()
                    stream_session.is_active = False
                    
                    # Update session stats
                    stats = self.client_stats.get(client_id, {})
                    stream_session.total_frames = stats.get('frames_processed', 0)
                    stream_session.total_detections = stats.get('detections_found', 0)
                    stream_session.avg_processing_time = stats.get('avg_processing_time', 0.0)
                    
                    session.commit()
                
        except Exception as e:
            logger.error(f"❌ Error ending stream session: {e}")
    
    def _broadcast_detection_result(self, client_id: str, result: Dict[str, Any]):
        """Broadcast detection result to desktop monitors"""
        try:
            if self.socketio:
                self.socketio.emit('frame_processed', {
                    'client_id': client_id,
                    'detections': result.get('detections', []),
                    'processing_time': result.get('processing_time', 0),
                    'frame_count': result.get('frame_count', 0),
                    'timestamp': time.time()
                }, room='desktop_monitors')
                
        except Exception as e:
            logger.error(f"❌ Error broadcasting result: {e}")
    
    def start_background_tasks(self):
        """Start background processing and cleanup tasks"""
        if self.running:
            return
        
        self.running = True
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name='StreamCleanup'
        )
        self.cleanup_thread.start()
        
        logger.info("✅ Background tasks started")
    
    def stop_background_tasks(self):
        """Stop background tasks"""
        self.running = False
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
        
        logger.info("✅ Background tasks stopped")
    
    def _cleanup_worker(self):
        """Background cleanup worker"""
        while self.running:
            try:
                current_time = time.time()
                
                # Clean up inactive clients
                inactive_clients = []
                for client_id, client_info in self.clients.items():
                    if current_time - client_info.get('last_activity', 0) > self.frame_timeout:
                        inactive_clients.append(client_id)
                
                for client_id in inactive_clients:
                    logger.info(f"🧹 Cleaning up inactive client: {client_id}")
                    self.unregister_client(client_id)
                    
                    # Notify desktop monitors
                    if self.socketio:
                        self.socketio.emit('client_timeout', {
                            'client_id': client_id,
                            'reason': 'inactivity',
                            'timestamp': time.time()
                        }, room='desktop_monitors')
                
                # Update cleanup timestamp
                self.global_stats['last_cleanup'] = current_time
                
                # Sleep until next cleanup
                time.sleep(self.config.STREAM_CLEANUP_INTERVAL)
                
            except Exception as e:
                logger.error(f"❌ Cleanup worker error: {e}")
                time.sleep(10)  # Wait before retrying
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics"""
        current_time = time.time()
        uptime = current_time - self.global_stats['service_start_time']
        
        return {
            'service_uptime': uptime,
            'uptime_formatted': self._format_uptime(uptime),
            'connected_clients': len(self.clients),
            'max_clients': self.max_clients,
            'client_utilization': round(len(self.clients) / self.max_clients * 100, 1),
            'global_stats': self.global_stats.copy(),
            'per_client_stats': {
                client_id: self._get_client_stats(client_id)
                for client_id in self.clients.keys()
            },
            'performance': {
                'avg_fps_all_clients': self._calculate_avg_fps(),
                'total_processing_time': sum(
                    stats['avg_processing_time'] * stats['frames_processed']
                    for stats in self.client_stats.values()
                ),
                'frames_per_second': (
                    self.global_stats['total_frames_processed'] / uptime
                    if uptime > 0 else 0
                )
            }
        }
    
    def _calculate_avg_fps(self) -> float:
        """Calculate average FPS across all clients"""
        if not self.client_stats:
            return 0.0
        
        fps_values = [stats['fps'] for stats in self.client_stats.values()]
        return round(sum(fps_values) / len(fps_values), 1) if fps_values else 0.0
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about specific client"""
        if client_id not in self.clients:
            return None
        
        client_info = self.clients[client_id].copy()
        client_info['stats'] = self._get_client_stats(client_id)
        
        return client_info
    
    def get_all_clients(self) -> List[Dict[str, Any]]:
        """Get information about all connected clients"""
        clients = []
        
        for client_id, client_info in self.clients.items():
            client_data = client_info.copy()
            client_data['id'] = client_id
            client_data['stats'] = self._get_client_stats(client_id)
            clients.append(client_data)
        
        return clients
    
    def force_disconnect_client(self, client_id: str, reason: str = "forced") -> bool:
        """Force disconnect a client"""
        try:
            if client_id not in self.clients:
                return False
            
            # Notify client
            if self.socketio:
                self.socketio.emit('force_disconnect', {
                    'reason': reason,
                    'timestamp': time.time()
                }, room=client_id)
            
            # Unregister client
            self.unregister_client(client_id)
            
            logger.info(f"✅ Force disconnected client {client_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error force disconnecting client: {e}")
            return False
    
    def test_service(self) -> Dict[str, Any]:
        """Test streaming service health"""
        try:
            # Create test frame
            test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
            test_frame.fill(128)  # Gray image
            
            # Encode as base64
            _, buffer = cv2.imencode('.jpg', test_frame)
            test_frame_data = base64.b64encode(buffer).decode('utf-8')
            
            # Test processing
            test_client_id = f"test_{uuid.uuid4().hex[:8]}"
            
            # Register test client
            test_client_info = {
                'type': 'test',
                'user_agent': 'StreamService Test',
                'screen_size': '200x200'
            }
            
            self.register_client(test_client_id, test_client_info)
            
            # Process test frame
            start_time = time.time()
            result = self.process_video_frame(test_client_id, test_frame_data, self.face_service)
            processing_time = time.time() - start_time
            
            # Clean up test client
            self.unregister_client(test_client_id)
            
            return {
                'status': 'healthy',
                'test_processing_time': processing_time,
                'test_result_success': result.get('success', False),
                'connected_clients': len(self.clients),
                'service_stats': self.get_stats(),
                'face_service_available': self.face_service is not None
            }
            
        except Exception as e:
            logger.error(f"❌ Stream service test failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'connected_clients': len(self.clients),
                'face_service_available': self.face_service is not None
            }