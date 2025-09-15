# 🚀 BACKEND REALTIME STREAM SERVICE
# File: backend/services/stream_service.py

import cv2
import base64
import numpy as np
import time
import threading
import queue
import uuid
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class StreamClient:
    """Client connection information"""
    client_id: str
    socket_id: str
    client_type: str  # 'mobile', 'desktop'
    user_agent: str
    screen_size: str
    connected_at: float
    last_frame_time: float
    frame_count: int
    is_active: bool

@dataclass
class FrameData:
    """Video frame data structure"""
    client_id: str
    frame: np.ndarray
    timestamp: float
    resolution: str
    processing_time: float = 0.0

class StreamService:
    """Realtime video streaming service"""
    
    def __init__(self, socketio, face_service=None):
        self.socketio = socketio
        self.face_service = face_service
        
        # Client management
        self.clients: Dict[str, StreamClient] = {}
        self.client_lock = threading.Lock()
        
        # Frame processing
        self.frame_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.is_processing = False
        
        # Detection settings
        self.detection_active = False
        self.detection_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'total_frames_received': 0,
            'total_frames_processed': 0,
            'total_detections': 0,
            'average_processing_time': 0.0,
            'fps': 0.0,
            'start_time': time.time(),
            'last_fps_update': time.time(),
            'frames_in_last_second': 0
        }
        
        # Start background processing
        self.start_processing()
        
        logger.info("✅ Stream service initialized")
    
    def start_processing(self):
        """Start background frame processing thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop, 
            daemon=True,
            name="StreamProcessor"
        )
        self.processing_thread.start()
        logger.info("🎬 Started frame processing thread")
    
    def stop_processing(self):
        """Stop background processing"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        logger.info("⏹️ Stopped frame processing")
    
    def _processing_loop(self):
        """Background frame processing loop"""
        while self.is_processing:
            try:
                # Get frame from queue (with timeout)
                frame_data = self.frame_queue.get(timeout=1.0)
                
                # Process frame
                self._process_frame(frame_data)
                
                # Mark task as done
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Frame processing error: {e}")
    
    def register_client(self, client_id: str, socket_id: str, client_info: Dict[str, Any]) -> bool:
        """Register new streaming client"""
        try:
            with self.client_lock:
                client = StreamClient(
                    client_id=client_id,
                    socket_id=socket_id,
                    client_type=client_info.get('type', 'unknown'),
                    user_agent=client_info.get('user_agent', ''),
                    screen_size=client_info.get('screen_size', ''),
                    connected_at=time.time(),
                    last_frame_time=0.0,
                    frame_count=0,
                    is_active=True
                )
                
                self.clients[client_id] = client
                
            logger.info(f"📱 Registered client: {client_id} ({client.client_type})")
            
            # Notify client
            if self.socketio:
                self.socketio.emit('mobile_registered', {
                    'client_id': client_id,
                    'detection_active': self.detection_active,
                    'server_time': time.time()
                }, to=socket_id)
            
            # Update all clients with stats
            self._broadcast_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Client registration failed: {e}")
            return False
    
    def unregister_client(self, client_id: str) -> bool:
        """Unregister client"""
        try:
            with self.client_lock:
                if client_id in self.clients:
                    client = self.clients.pop(client_id)
                    logger.info(f"📱 Unregistered client: {client_id}")
                    
                    # Update stats
                    self._broadcast_stats()
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"❌ Client unregistration failed: {e}")
            return False
    
    def process_video_frame(self, client_id: str, frame_data: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process incoming video frame from mobile"""
        start_time = time.time()
        
        try:
            # Validate client
            if client_id not in self.clients:
                return {'success': False, 'error': 'Client not registered'}
            
            # Decode base64 frame
            try:
                # Remove data URL prefix if present
                if ',' in frame_data:
                    frame_data = frame_data.split(',')[1]
                
                # Decode base64 to bytes
                img_bytes = base64.b64decode(frame_data)
                
                # Convert to numpy array
                nparr = np.frombuffer(img_bytes, np.uint8)
                
                # Decode image
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    return {'success': False, 'error': 'Failed to decode frame'}
                
            except Exception as e:
                return {'success': False, 'error': f'Frame decode error: {str(e)}'}
            
            # Update client stats
            with self.client_lock:
                client = self.clients[client_id]
                client.last_frame_time = time.time()
                client.frame_count += 1
            
            # Update global stats
            self.stats['total_frames_received'] += 1
            self._update_fps()
            
            # Create frame data object
            frame_obj = FrameData(
                client_id=client_id,
                frame=frame,
                timestamp=time.time(),
                resolution=f"{frame.shape[1]}x{frame.shape[0]}"
            )
            
            # Add to processing queue if detection is active
            if self.detection_active and not self.frame_queue.full():
                try:
                    self.frame_queue.put_nowait(frame_obj)
                except queue.Full:
                    logger.warning("⚠️ Frame queue full, dropping frame")
            
            # Broadcast frame to desktop monitors
            self._broadcast_frame_to_desktop(client_id, frame_data, metadata)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'processing_time': processing_time,
                'frame_resolution': frame_obj.resolution,
                'detection_active': self.detection_active,
                'queue_size': self.frame_queue.qsize()
            }
            
        except Exception as e:
            logger.error(f"❌ Frame processing error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_frame(self, frame_data: FrameData):
        """Process frame for face detection"""
        if not self.face_service:
            return
        
        start_time = time.time()
        
        try:
            # Perform face detection/recognition
            detection_results = self.face_service.detect_and_recognize_faces(frame_data.frame)
            
            processing_time = time.time() - start_time
            frame_data.processing_time = processing_time
            
            # Update stats
            self.stats['total_frames_processed'] += 1
            self.stats['total_detections'] += len(detection_results)
            
            # Update average processing time
            total_processed = self.stats['total_frames_processed']
            self.stats['average_processing_time'] = (
                (self.stats['average_processing_time'] * (total_processed - 1) + processing_time) 
                / total_processed
            )
            
            # Send results back to mobile client
            self._send_detection_results(frame_data.client_id, detection_results, processing_time)
            
            # Broadcast to desktop monitors
            self._broadcast_detection_to_desktop(frame_data.client_id, detection_results)
            
        except Exception as e:
            logger.error(f"❌ Face detection error: {e}")
    
    def _send_detection_results(self, client_id: str, results: List[Dict], processing_time: float):
        """Send detection results to mobile client"""
        try:
            if client_id not in self.clients:
                return
            
            client = self.clients[client_id]
            
            # Format results for mobile display
            formatted_faces = []
            for result in results:
                face_data = {
                    'x': result.get('x', 0),
                    'y': result.get('y', 0),
                    'width': result.get('width', 0),
                    'height': result.get('height', 0),
                    'confidence': result.get('confidence', 0.0)
                }
                
                # Add employee info if recognized
                if result.get('employee'):
                    face_data['employee'] = {
                        'id': result['employee'].get('id'),
                        'name': result['employee'].get('name'),
                        'employee_code': result['employee'].get('employee_code'),
                        'department': result['employee'].get('department')
                    }
                
                formatted_faces.append(face_data)
            
            # Send to mobile
            self.socketio.emit('detection_result', {
                'success': True,
                'faces': formatted_faces,
                'processing_time': processing_time,
                'timestamp': time.time()
            }, to=client.socket_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to send detection results: {e}")
    
    def _broadcast_frame_to_desktop(self, client_id: str, frame_data: str, metadata: Dict = None):
        """Broadcast frame to desktop monitoring clients"""
        try:
            # Find desktop clients
            desktop_clients = [
                client for client in self.clients.values() 
                if client.client_type == 'desktop' and client.is_active
            ]
            
            if not desktop_clients:
                return
            
            # Prepare frame data for desktop
            frame_info = {
                'client_id': client_id,
                'frame_data': frame_data,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            
            # Broadcast to all desktop clients
            for desktop_client in desktop_clients:
                self.socketio.emit('mobile_frame_received', frame_info, to=desktop_client.socket_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to broadcast frame to desktop: {e}")
    
    def _broadcast_detection_to_desktop(self, client_id: str, detection_results: List[Dict]):
        """Broadcast detection results to desktop clients"""
        try:
            desktop_clients = [
                client for client in self.clients.values() 
                if client.client_type == 'desktop' and client.is_active
            ]
            
            if not desktop_clients:
                return
            
            detection_info = {
                'client_id': client_id,
                'detections': detection_results,
                'timestamp': time.time(),
                'detection_count': len(detection_results)
            }
            
            for desktop_client in desktop_clients:
                self.socketio.emit('detection_broadcast', detection_info, to=desktop_client.socket_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to broadcast detection to desktop: {e}")
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.stats['frames_in_last_second'] += 1
        
        # Update FPS every second
        if current_time - self.stats['last_fps_update'] >= 1.0:
            self.stats['fps'] = self.stats['frames_in_last_second']
            self.stats['frames_in_last_second'] = 0
            self.stats['last_fps_update'] = current_time
    
    def _broadcast_stats(self):
        """Broadcast current statistics to all clients"""
        try:
            stats_data = self.get_current_stats()
            
            # Send to all connected clients
            for client in self.clients.values():
                if client.is_active:
                    self.socketio.emit('server_stats', stats_data, to=client.socket_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to broadcast stats: {e}")
    
    def toggle_detection(self) -> bool:
        """Toggle face detection on/off"""
        with self.detection_lock:
            self.detection_active = not self.detection_active
            
        logger.info(f"🔍 Detection {'enabled' if self.detection_active else 'disabled'}")
        
        # Notify all clients
        for client in self.clients.values():
            if client.is_active:
                self.socketio.emit('detection_toggled', {
                    'detection_active': self.detection_active,
                    'timestamp': time.time()
                }, to=client.socket_id)
        
        return self.detection_active
    
    def set_detection_active(self, active: bool) -> bool:
        """Set detection state"""
        with self.detection_lock:
            self.detection_active = active
            
        logger.info(f"🔍 Detection set to {'enabled' if active else 'disabled'}")
        
        # Notify all clients
        for client in self.clients.values():
            if client.is_active:
                self.socketio.emit('detection_toggled', {
                    'detection_active': self.detection_active,
                    'timestamp': time.time()
                }, to=client.socket_id)
        
        return self.detection_active
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current streaming statistics"""
        current_time = time.time()
        uptime = current_time - self.stats['start_time']
        
        # Client statistics
        client_stats = {
            'total_clients': len(self.clients),
            'mobile_clients': len([c for c in self.clients.values() if c.client_type == 'mobile']),
            'desktop_clients': len([c for c in self.clients.values() if c.client_type == 'desktop']),
            'active_clients': len([c for c in self.clients.values() if c.is_active])
        }
        
        # Processing statistics
        processing_stats = {
            'total_frames_received': self.stats['total_frames_received'],
            'total_frames_processed': self.stats['total_frames_processed'],
            'total_detections': self.stats['total_detections'],
            'average_processing_time': round(self.stats['average_processing_time'], 4),
            'current_fps': self.stats['fps'],
            'queue_size': self.frame_queue.qsize(),
            'detection_active': self.detection_active
        }
        
        # System statistics
        system_stats = {
            'uptime': round(uptime, 2),
            'uptime_formatted': self._format_uptime(uptime),
            'face_service_available': self.face_service is not None,
            'processing_thread_alive': self.processing_thread and self.processing_thread.is_alive()
        }
        
        return {
            'clients': client_stats,
            'processing': processing_stats,
            'system': system_stats,
            'timestamp': current_time
        }
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get specific client information"""
        if client_id not in self.clients:
            return None
            
        client = self.clients[client_id]
        current_time = time.time()
        
        return {
            'client_id': client.client_id,
            'client_type': client.client_type,
            'user_agent': client.user_agent,
            'screen_size': client.screen_size,
            'connected_at': client.connected_at,
            'connection_duration': round(current_time - client.connected_at, 2),
            'last_frame_time': client.last_frame_time,
            'frame_count': client.frame_count,
            'is_active': client.is_active,
            'time_since_last_frame': round(current_time - client.last_frame_time, 2) if client.last_frame_time > 0 else 0
        }
    
    def cleanup_inactive_clients(self, timeout_seconds: int = 300) -> int:
        """Clean up inactive clients (5 minutes timeout by default)"""
        current_time = time.time()
        removed_count = 0
        
        with self.client_lock:
            inactive_clients = []
            
            for client_id, client in self.clients.items():
                # Check if client is inactive
                if (current_time - client.last_frame_time > timeout_seconds and 
                    client.last_frame_time > 0):
                    inactive_clients.append(client_id)
            
            # Remove inactive clients
            for client_id in inactive_clients:
                client = self.clients.pop(client_id)
                logger.info(f"🧹 Removed inactive client: {client_id}")
                removed_count += 1
        
        if removed_count > 0:
            self._broadcast_stats()
        
        return removed_count
    
    def force_disconnect_client(self, client_id: str, reason: str = "Admin disconnect") -> bool:
        """Force disconnect a specific client"""
        try:
            if client_id not in self.clients:
                return False
            
            client = self.clients[client_id]
            
            # Notify client
            if self.socketio:
                self.socketio.emit('force_disconnect', {
                    'reason': reason,
                    'timestamp': time.time()
                }, to=client.socket_id)
            
            # Remove client
            self.unregister_client(client_id)
            
            logger.info(f"✅ Force disconnected client {client_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error force disconnecting client: {e}")
            return False
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            current_time = time.time()
            
            # Check processing thread
            thread_healthy = self.processing_thread and self.processing_thread.is_alive()
            
            # Check queue status
            queue_healthy = self.frame_queue.qsize() < self.frame_queue.maxsize * 0.8
            
            # Check recent activity
            recent_activity = any(
                current_time - client.last_frame_time < 60 
                for client in self.clients.values() 
                if client.last_frame_time > 0
            )
            
            # Overall health
            is_healthy = thread_healthy and queue_healthy
            
            return {
                'status': 'healthy' if is_healthy else 'degraded',
                'checks': {
                    'processing_thread': thread_healthy,
                    'queue_status': queue_healthy,
                    'recent_activity': recent_activity,
                    'face_service': self.face_service is not None
                },
                'stats': self.get_current_stats(),
                'timestamp': current_time
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down stream service...")
        
        # Stop processing
        self.stop_processing()
        
        # Disconnect all clients
        with self.client_lock:
            for client_id in list(self.clients.keys()):
                self.force_disconnect_client(client_id, "Server shutdown")
        
        logger.info("✅ Stream service shutdown complete")