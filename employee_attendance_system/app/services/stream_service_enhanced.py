#!/usr/bin/env python3
"""
stream_service.py - REAL-TIME VIDEO STREAM SERVICE
Handles continuous video streaming from mobile devices to desktop
Optimized for low-latency face recognition and real-time feedback
"""

import time
import threading
import queue
import cv2
import numpy as np
import base64
import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class ClientInfo:
    """Information about connected streaming client"""
    client_id: str
    client_type: str  # 'mobile' or 'desktop'
    user_agent: str
    connected_at: float
    last_frame_time: float
    frames_received: int
    total_bytes_received: int
    fps: float
    active: bool = True

@dataclass
class FrameData:
    """Video frame data with metadata"""
    client_id: str
    frame: np.ndarray
    timestamp: float
    sequence_number: int
    quality: float

class FPSCalculator:
    """Calculate FPS for video streams"""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        
    def add_frame(self, timestamp: float = None):
        """Add frame timestamp for FPS calculation"""
        if timestamp is None:
            timestamp = time.time()
        self.frame_times.append(timestamp)
    
    def get_fps(self) -> float:
        """Calculate current FPS"""
        if len(self.frame_times) < 2:
            return 0.0
        
        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span <= 0:
            return 0.0
        
        return (len(self.frame_times) - 1) / time_span

class StreamProcessor:
    """Process video streams with face recognition"""
    
    def __init__(self, socketio, face_service):
        self.socketio = socketio
        self.face_service = face_service
        self.processing_queue = queue.Queue(maxsize=100)
        self.results_cache = {}
        self.processing_thread = None
        self.running = False
        
    def start(self):
        """Start stream processing thread"""
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
            self.processing_thread.start()
            logger.info("✅ Stream processor started")
    
    def stop(self):
        """Stop stream processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        logger.info("🛑 Stream processor stopped")
    
    def add_frame(self, frame_data: FrameData):
        """Add frame to processing queue"""
        try:
            self.processing_queue.put(frame_data, block=False)
        except queue.Full:
            logger.warning("⚠️ Processing queue full, dropping frame")
    
    def _process_frames(self):
        """Main frame processing loop"""
        while self.running:
            try:
                frame_data = self.processing_queue.get(timeout=1.0)
                
                # Process frame with face recognition
                start_time = time.time()
                detections = self.face_service.recognize_face(frame_data.frame)
                processing_time = time.time() - start_time
                
                # Prepare result
                result = {
                    'success': True,
                    'timestamp': time.time(),
                    'client_timestamp': frame_data.timestamp,
                    'sequence_number': frame_data.sequence_number,
                    'detections': detections,
                    'processing_time_ms': round(processing_time * 1000, 2),
                    'frame_size': frame_data.frame.shape if frame_data.frame is not None else None
                }
                
                # Send result back to client
                self.socketio.emit('detection_result', result, room=frame_data.client_id)
                
                # Broadcast to desktop monitors
                self.socketio.emit('frame_processed', {
                    'client_id': frame_data.client_id,
                    'detections_count': len(detections),
                    'processing_time_ms': result['processing_time_ms'],
                    'timestamp': time.time()
                }, room='desktop_monitors')
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Frame processing error: {e}")

class StreamService:
    """Real-time video streaming service"""
    
    def __init__(self, socketio, face_service):
        self.socketio = socketio
        self.face_service = face_service
        self.clients = {}  # client_id -> ClientInfo
        self.fps_calculators = {}  # client_id -> FPSCalculator
        self.stats = {
            'total_frames': 0,
            'total_bytes': 0,
            'total_detections': 0,
            'start_time': time.time(),
            'processing_times': deque(maxlen=100)
        }
        
        # Stream processor for face recognition
        self.processor = StreamProcessor(socketio, face_service)
        
        # Settings
        self.max_clients = 10
        self.frame_timeout = 30  # seconds
        self.cleanup_interval = 60  # seconds
        self.max_frame_size = 2 * 1024 * 1024  # 2MB
        
        # Background tasks
        self.cleanup_thread = None
        self.stats_thread = None
        self.running = False
        
        logger.info("✅ StreamService initialized")
    
    def start_background_tasks(self):
        """Start background tasks"""
        if not self.running:
            self.running = True
            
            # Start stream processor
            self.processor.start()
            
            # Start cleanup task
            self.cleanup_thread = threading.Thread(target=self._cleanup_task, daemon=True)
            self.cleanup_thread.start()
            
            # Start stats broadcasting task
            self.stats_thread = threading.Thread(target=self._stats_task, daemon=True)
            self.stats_thread.start()
            
            logger.info("✅ StreamService background tasks started")
    
    def stop_background_tasks(self):
        """Stop background tasks"""
        self.running = False
        self.processor.stop()
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        if self.stats_thread:
            self.stats_thread.join(timeout=5.0)
            
        logger.info("🛑 StreamService background tasks stopped")
    
    def register_client(self, client_id: str, client_info: Dict) -> int:
        """Register new streaming client"""
        try:
            if len(self.clients) >= self.max_clients:
                logger.warning(f"⚠️ Max clients ({self.max_clients}) reached")
                return len(self.clients)
            
            # Create client info
            client = ClientInfo(
                client_id=client_id,
                client_type=client_info.get('type', 'unknown'),
                user_agent=client_info.get('user_agent', ''),
                connected_at=time.time(),
                last_frame_time=0,
                frames_received=0,
                total_bytes_received=0,
                fps=0.0
            )
            
            self.clients[client_id] = client
            self.fps_calculators[client_id] = FPSCalculator()
            
            logger.info(f"📱 Client registered: {client_id[:8]}... ({client.client_type})")
            return len(self.clients)
            
        except Exception as e:
            logger.error(f"❌ Error registering client: {e}")
            return len(self.clients)
    
    def unregister_client(self, client_id: str) -> int:
        """Unregister streaming client"""
        try:
            if client_id in self.clients:
                client = self.clients[client_id]
                del self.clients[client_id]
                
                if client_id in self.fps_calculators:
                    del self.fps_calculators[client_id]
                
                logger.info(f"📱 Client unregistered: {client_id[:8]}... ({client.client_type})")
            
            return len(self.clients)
            
        except Exception as e:
            logger.error(f"❌ Error unregistering client: {e}")
            return len(self.clients)
    
    def process_video_frame(self, client_id: str, frame_data: str, face_service) -> Dict:
        """Process video frame from mobile client"""
        try:
            start_time = time.time()
            
            # Validate client
            if client_id not in self.clients:
                return {'success': False, 'error': 'Client not registered'}
            
            client = self.clients[client_id]
            
            # Validate frame data
            if not frame_data:
                return {'success': False, 'error': 'No frame data'}
            
            # Check frame size
            frame_size = len(frame_data)
            if frame_size > self.max_frame_size:
                return {'success': False, 'error': 'Frame too large'}
            
            # Decode frame
            frame = self._decode_frame(frame_data)
            if frame is None:
                return {'success': False, 'error': 'Could not decode frame'}
            
            # Update client stats
            current_time = time.time()
            client.frames_received += 1
            client.last_frame_time = current_time
            client.total_bytes_received += frame_size
            
            # Update FPS
            fps_calc = self.fps_calculators[client_id]
            fps_calc.add_frame(current_time)
            client.fps = fps_calc.get_fps()
            
            # Update global stats
            self.stats['total_frames'] += 1
            self.stats['total_bytes'] += frame_size
            
            # Process with face recognition (if available)
            detections = []
            processing_time = 0
            
            if face_service:
                recognition_start = time.time()
                detections = face_service.recognize_face(frame)
                processing_time = time.time() - recognition_start
                
                self.stats['total_detections'] += len(detections)
                self.stats['processing_times'].append(processing_time)
            
            # Prepare result
            result = {
                'success': True,
                'timestamp': current_time,
                'detections': detections,
                'processing_time_ms': round(processing_time * 1000, 2),
                'frame_stats': {
                    'frame_number': client.frames_received,
                    'frame_size_bytes': frame_size,
                    'fps': round(client.fps, 1)
                },
                'client_id': client_id
            }
            
            # Record total processing time
            total_processing_time = time.time() - start_time
            logger.debug(f"🎬 Processed frame {client.frames_received} from {client_id[:8]}... in {total_processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Frame processing error for {client_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _decode_frame(self, frame_data: str) -> Optional[np.ndarray]:
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
            logger.error(f"❌ Frame decode error: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Get comprehensive streaming statistics"""
        current_time = time.time()
        uptime = current_time - self.stats['start_time']
        
        # Calculate average processing time
        avg_processing_time = 0
        if self.stats['processing_times']:
            avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
        
        # Calculate overall FPS
        overall_fps = 0
        if uptime > 0:
            overall_fps = self.stats['total_frames'] / uptime
        
        # Get active clients
        active_clients = [
            {
                'id': client_id[:8] + '...',
                'type': client.client_type,
                'connected_duration': current_time - client.connected_at,
                'frames_received': client.frames_received,
                'fps': round(client.fps, 1),
                'total_mb': round(client.total_bytes_received / (1024 * 1024), 2),
                'last_frame_ago': current_time - client.last_frame_time if client.last_frame_time > 0 else 0
            }
            for client_id, client in self.clients.items()
        ]
        
        return {
            'connected_clients': len(self.clients),
            'active_clients': active_clients,
            'total_frames': self.stats['total_frames'],
            'total_detections': self.stats['total_detections'],
            'total_mb_processed': round(self.stats['total_bytes'] / (1024 * 1024), 2),
            'uptime_seconds': uptime,
            'uptime_formatted': self._format_duration(uptime),
            'overall_fps': round(overall_fps, 2),
            'avg_processing_time_ms': round(avg_processing_time * 1000, 2),
            'queue_size': self.processor.processing_queue.qsize() if hasattr(self.processor, 'processing_queue') else 0,
            'system_info': {
                'max_clients': self.max_clients,
                'frame_timeout': self.frame_timeout,
                'max_frame_size_mb': round(self.max_frame_size / (1024 * 1024), 2)
            }
        }
    
    def get_client_info(self, client_id: str) -> Optional[Dict]:
        """Get detailed information about specific client"""
        if client_id not in self.clients:
            return None
        
        client = self.clients[client_id]
        current_time = time.time()
        
        return {
            'client_id': client_id,
            'type': client.client_type,
            'user_agent': client.user_agent,
            'connected_at': client.connected_at,
            'connected_duration': current_time - client.connected_at,
            'last_frame_time': client.last_frame_time,
            'time_since_last_frame': current_time - client.last_frame_time if client.last_frame_time > 0 else 0,
            'frames_received': client.frames_received,
            'total_bytes_received': client.total_bytes_received,
            'total_mb_received': round(client.total_bytes_received / (1024 * 1024), 2),
            'current_fps': round(client.fps, 1),
            'active': client.active
        }
    
    def cleanup_inactive_clients(self):
        """Remove inactive clients"""
        current_time = time.time()
        inactive_clients = []
        
        for client_id, client in self.clients.items():
            time_since_last_frame = current_time - client.last_frame_time
            
            # Mark as inactive if no frames for timeout period
            if client.last_frame_time > 0 and time_since_last_frame > self.frame_timeout:
                inactive_clients.append(client_id)
                client.active = False
        
        # Remove inactive clients
        for client_id in inactive_clients:
            self.unregister_client(client_id)
            logger.info(f"🧹 Removed inactive client: {client_id[:8]}...")
        
        return len(inactive_clients)
    
    def _cleanup_task(self):
        """Background task for cleaning up inactive clients"""
        while self.running:
            try:
                removed_count = self.cleanup_inactive_clients()
                if removed_count > 0:
                    logger.info(f"🧹 Cleaned up {removed_count} inactive clients")
                
                time.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"❌ Cleanup task error: {e}")
                time.sleep(self.cleanup_interval)
    
    def _stats_task(self):
        """Background task for broadcasting statistics"""
        while self.running:
            try:
                # Broadcast stats to desktop monitors
                stats = self.get_stats()
                self.socketio.emit('stats_update', {
                    'stats': stats,
                    'timestamp': time.time()
                }, room='desktop_monitors')
                
                time.sleep(5)  # Broadcast every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Stats broadcasting error: {e}")
                time.sleep(5)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def set_detection_enabled(self, enabled: bool):
        """Enable/disable face detection (placeholder for compatibility)"""
        logger.info(f"🔍 Face detection: {'ENABLED' if enabled else 'DISABLED'}")
        
        # Broadcast status change
        self.socketio.emit('detection_status_changed', {
            'active': enabled,
            'timestamp': time.time()
        })
    
    def is_detection_enabled(self) -> bool:
        """Check if face detection is enabled (placeholder)"""
        return True  # Always enabled if face_service is available
    
    def export_session_data(self) -> Dict:
        """Export session data for analysis"""
        return {
            'session_info': {
                'start_time': self.stats['start_time'],
                'current_time': time.time(),
                'duration': time.time() - self.stats['start_time']
            },
            'stats': self.get_stats(),
            'clients': [
                self.get_client_info(client_id)
                for client_id in self.clients.keys()
            ],
            'system_settings': {
                'max_clients': self.max_clients,
                'frame_timeout': self.frame_timeout,
                'max_frame_size': self.max_frame_size,
                'cleanup_interval': self.cleanup_interval
            }
        }
    
    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            'total_frames': 0,
            'total_bytes': 0,
            'total_detections': 0,
            'start_time': time.time(),
            'processing_times': deque(maxlen=100)
        }
        
        # Reset client stats
        for client in self.clients.values():
            client.frames_received = 0
            client.total_bytes_received = 0
            client.connected_at = time.time()
        
        # Reset FPS calculators
        for fps_calc in self.fps_calculators.values():
            fps_calc.frame_times.clear()
        
        logger.info("📊 Statistics reset")
    
    def get_network_info(self) -> Dict:
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
                'mobile_url': f'http://{local_ip}:5000/mobile',
                'desktop_url': f'http://{local_ip}:5000',
                'websocket_url': f'ws://{local_ip}:5000/socket.io/'
            }
            
        except Exception as e:
            logger.error(f"❌ Could not get network info: {e}")
            return {
                'local_ip': 'localhost',
                'port': 5000,
                'mobile_url': 'http://localhost:5000/mobile',
                'desktop_url': 'http://localhost:5000',
                'error': str(e)
            }
    
    def get_performance_metrics(self) -> Dict:
        """Get detailed performance metrics"""
        stats = self.get_stats()
        
        # Add performance-specific metrics
        performance_data = {
            'memory_usage': self._get_memory_usage(),
            'processing_queue_size': self.processor.processing_queue.qsize() if hasattr(self.processor, 'processing_queue') else 0,
            'active_threads': threading.active_count(),
            'frame_processing_rate': stats.get('overall_fps', 0),
            'detection_success_rate': self._calculate_detection_success_rate(),
            'average_latency_ms': stats.get('avg_processing_time_ms', 0),
            'network_throughput_mbps': self._calculate_network_throughput()
        }
        
        return {**stats, **performance_data}
    
    def _get_memory_usage(self) -> Dict:
        """Get memory usage information"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': round(memory_info.rss / (1024 * 1024), 2),
                'vms_mb': round(memory_info.vms / (1024 * 1024), 2),
                'percent': round(process.memory_percent(), 2)
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_detection_success_rate(self) -> float:
        """Calculate face detection success rate"""
        if self.stats['total_frames'] == 0:
            return 0.0
        
        return round((self.stats['total_detections'] / self.stats['total_frames']) * 100, 2)
    
    def _calculate_network_throughput(self) -> float:
        """Calculate network throughput in Mbps"""
        current_time = time.time()
        uptime = current_time - self.stats['start_time']
        
        if uptime <= 0:
            return 0.0
        
        total_bits = self.stats['total_bytes'] * 8
        throughput_bps = total_bits / uptime
        throughput_mbps = throughput_bps / (1024 * 1024)
        
        return round(throughput_mbps, 2)
    
    def update_settings(self, new_settings: Dict):
        """Update stream service settings"""
        if 'max_clients' in new_settings:
            self.max_clients = int(new_settings['max_clients'])
        
        if 'frame_timeout' in new_settings:
            self.frame_timeout = int(new_settings['frame_timeout'])
        
        if 'cleanup_interval' in new_settings:
            self.cleanup_interval = int(new_settings['cleanup_interval'])
        
        if 'max_frame_size' in new_settings:
            self.max_frame_size = int(new_settings['max_frame_size'])
        
        logger.info(f"⚙️ Updated stream settings: {new_settings}")
    
    def __del__(self):
        """Cleanup when service is destroyed"""
        try:
            self.stop_background_tasks()
        except:
            pass