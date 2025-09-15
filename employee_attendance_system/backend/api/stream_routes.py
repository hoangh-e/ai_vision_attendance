#!/usr/bin/env python3
"""
stream_routes.py - VIDEO STREAMING API ROUTES
API endpoints for real-time video streaming and face detection
"""

from flask import Blueprint, request, jsonify
from flask_socketio import emit
import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Create blueprint
stream_bp = Blueprint('stream', __name__, url_prefix='/api/stream')

# This will be set by main_app.py
stream_service = None
face_service = None

def init_stream_routes(stream_svc, face_svc):
    """Initialize the routes with service instances"""
    global stream_service, face_service
    stream_service = stream_svc
    face_service = face_svc

@stream_bp.route('/status', methods=['GET'])
def get_stream_status():
    """
    Get current streaming status and statistics
    """
    try:
        if not stream_service:
            return jsonify({'success': False, 'error': 'Stream service not initialized'})
        
        stats = stream_service.get_current_stats()
        clients = stream_service.get_all_clients()
        
        return jsonify({
            'success': True,
            'data': {
                'stats': stats,
                'connected_clients': len(clients),
                'clients': [
                    {
                        'id': client_id,
                        'type': client_info.get('type', 'unknown'),
                        'connected_at': client_info.get('connected_at'),
                        'frames_sent': client_info.get('frames_sent', 0),
                        'user_agent': client_info.get('user_agent', '')
                    }
                    for client_id, client_info in clients.items()
                ]
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting stream status: {e}")
        return jsonify({'success': False, 'error': str(e)})

@stream_bp.route('/clients', methods=['GET'])
def get_connected_clients():
    """
    Get list of connected streaming clients
    """
    try:
        if not stream_service:
            return jsonify({'success': False, 'error': 'Stream service not initialized'})
        
        clients = stream_service.get_all_clients()
        
        client_list = []
        for client_id, client_info in clients.items():
            client_data = {
                'id': client_id,
                'type': client_info.get('type', 'unknown'),
                'connected_at': client_info.get('connected_at'),
                'frames_sent': client_info.get('frames_sent', 0),
                'last_frame_time': client_info.get('last_frame_time'),
                'active': client_info.get('active', False)
            }
            
            # Calculate connection duration
            if client_info.get('connected_at'):
                client_data['connection_duration_seconds'] = time.time() - client_info['connected_at']
            
            client_list.append(client_data)
        
        return jsonify({
            'success': True,
            'data': {
                'total_clients': len(client_list),
                'clients': client_list
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting connected clients: {e}")
        return jsonify({'success': False, 'error': str(e)})

@stream_bp.route('/client/<client_id>', methods=['GET'])
def get_client_info(client_id: str):
    """
    Get detailed information about a specific client
    """
    try:
        if not stream_service:
            return jsonify({'success': False, 'error': 'Stream service not initialized'})
        
        client_info = stream_service.get_client_info(client_id)
        
        if not client_info:
            return jsonify({'success': False, 'error': 'Client not found'})
        
        # Calculate additional metrics
        response_data = dict(client_info)
        if client_info.get('connected_at'):
            response_data['connection_duration_seconds'] = time.time() - client_info['connected_at']
        
        if client_info.get('last_frame_time'):
            response_data['time_since_last_frame'] = time.time() - client_info['last_frame_time']
        
        return jsonify({
            'success': True,
            'data': response_data
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting client info: {e}")
        return jsonify({'success': False, 'error': str(e)})

@stream_bp.route('/settings', methods=['GET'])
def get_stream_settings():
    """
    Get current streaming and detection settings
    """
    try:
        if not face_service:
            return jsonify({'success': False, 'error': 'Face service not initialized'})
        
        settings = {
            'face_recognition': {
                'threshold': face_service.threshold,
                'detection_backend': face_service.detection_backend,
                'model_name': face_service.model_name,
                'max_face_size': face_service.max_face_size
            },
            'streaming': {
                'max_clients': 10,  # Can be made configurable
                'frame_timeout_seconds': 30,
                'cleanup_interval_seconds': 60
            }
        }
        
        return jsonify({
            'success': True,
            'data': settings
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting stream settings: {e}")
        return jsonify({'success': False, 'error': str(e)})

@stream_bp.route('/settings/threshold', methods=['POST'])
def update_recognition_threshold():
    """
    Update face recognition threshold
    """
    try:
        if not face_service:
            return jsonify({'success': False, 'error': 'Face service not initialized'})
        
        data = request.get_json()
        new_threshold = data.get('threshold')
        
        if new_threshold is None:
            return jsonify({'success': False, 'error': 'Threshold value required'})
        
        if not isinstance(new_threshold, (int, float)) or not (0.0 <= new_threshold <= 1.0):
            return jsonify({'success': False, 'error': 'Threshold must be a number between 0.0 and 1.0'})
        
        face_service.update_threshold(float(new_threshold))
        
        return jsonify({
            'success': True,
            'data': {
                'threshold': face_service.threshold,
                'message': f'Recognition threshold updated to {face_service.threshold}'
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error updating threshold: {e}")
        return jsonify({'success': False, 'error': str(e)})

@stream_bp.route('/performance', methods=['GET'])
def get_performance_metrics():
    """
    Get detailed performance metrics
    """
    try:
        if not stream_service:
            return jsonify({'success': False, 'error': 'Stream service not initialized'})
        
        stats = stream_service.get_current_stats()
        
        # Add face recognition stats if available
        if face_service:
            recognition_stats = face_service.get_recognition_stats()
            stats['face_recognition'] = recognition_stats
        
        return jsonify({
            'success': True,
            'data': stats
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting performance metrics: {e}")
        return jsonify({'success': False, 'error': str(e)})

@stream_bp.route('/test/frame', methods=['POST'])
def test_frame_processing():
    """
    Test endpoint for frame processing (for debugging)
    """
    try:
        if not stream_service or not face_service:
            return jsonify({'success': False, 'error': 'Services not initialized'})
        
        # This is for testing purposes - in real app, frames come via SocketIO
        data = request.get_json()
        
        if not data or 'frame' not in data:
            return jsonify({'success': False, 'error': 'Frame data required'})
        
        # Process the frame
        result = stream_service.process_video_frame('test_client', data)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"❌ Error in test frame processing: {e}")
        return jsonify({'success': False, 'error': str(e)})

@stream_bp.route('/cleanup', methods=['POST'])
def cleanup_inactive_clients():
    """
    Manually trigger cleanup of inactive clients
    """
    try:
        if not stream_service:
            return jsonify({'success': False, 'error': 'Stream service not initialized'})
        
        clients_before = len(stream_service.get_all_clients())
        stream_service.cleanup_inactive_clients()
        clients_after = len(stream_service.get_all_clients())
        
        cleaned_up = clients_before - clients_after
        
        return jsonify({
            'success': True,
            'data': {
                'clients_before': clients_before,
                'clients_after': clients_after,
                'cleaned_up': cleaned_up,
                'message': f'Cleaned up {cleaned_up} inactive clients'
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error cleaning up clients: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Health check endpoint
@stream_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for streaming services
    """
    try:
        health_data = {
            'stream_service': stream_service is not None,
            'face_service': face_service is not None,
            'timestamp': time.time()
        }
        
        if stream_service:
            stats = stream_service.get_current_stats()
            health_data['connected_clients'] = stats.get('connected_clients', 0)
            health_data['total_frames_processed'] = stats.get('total_frames_processed', 0)
        
        return jsonify({
            'success': True,
            'data': health_data
        })
        
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Error handlers
@stream_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Stream endpoint not found'}), 404

@stream_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal stream service error'}), 500