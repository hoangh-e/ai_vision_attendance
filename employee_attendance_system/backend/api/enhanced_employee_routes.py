#!/usr/bin/env python3
"""
enhanced_employee_routes.py - EMPLOYEE MANAGEMENT API WITH IMAGE VECTOR PROCESSING
File: backend/api/enhanced_employee_routes.py
"""

from flask import Blueprint, request, jsonify, current_app
import logging
from typing import List, Dict, Any
import os
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

# Create blueprint
employee_bp = Blueprint('employee', __name__, url_prefix='/api/employees')

# Services will be injected by main app
employee_service = None
face_service = None

def init_employee_routes(emp_service, face_svc):
    """Initialize routes with service instances"""
    global employee_service, face_service
    employee_service = emp_service
    face_service = face_svc

@employee_bp.route('', methods=['GET'])
def get_all_employees():
    """
    Get all employees with their face image counts
    """
    try:
        if not employee_service:
            return jsonify({'success': False, 'error': 'Employee service not initialized'}), 500
        
        employees = employee_service.get_all_employees()
        
        result = []
        for emp in employees:
            employee_data = {
                'id': emp['id'],
                'name': emp['name'],
                'employee_code': emp['employee_code'],
                'department': emp.get('department'),
                'position': emp.get('position'),
                'email': emp.get('email'),
                'phone': emp.get('phone'),
                'status': emp.get('status', 'active'),
                'created_at': emp.get('created_at'),
                'image_count': 0
            }
            
            # Get face image count using enhanced face service
            if face_service:
                employee_data['image_count'] = face_service.get_employee_image_count(emp['id'])
            
            result.append(employee_data)
        
        return jsonify({
            'success': True,
            'data': result,
            'total': len(result),
            'timestamp': current_app._get_current_object().config.get('CURRENT_TIME', 'N/A')
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting employees: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@employee_bp.route('', methods=['POST'])
def create_employee():
    """
    Create a new employee
    """
    try:
        if not employee_service:
            return jsonify({'success': False, 'error': 'Employee service not initialized'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['name', 'employee_code']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'error': f'Field {field} is required'}), 400
        
        # Create employee using enhanced service
        employee = employee_service.create_employee(**data)
        
        if employee:
            return jsonify({
                'success': True,
                'data': {
                    'id': employee.id,
                    'name': data.get('name'),
                    'employee_code': data.get('employee_code'),
                    'department': data.get('department'),
                    'image_count': 0
                },
                'message': f'Employee {data.get("name")} created successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to create employee'}), 500
            
    except Exception as e:
        logger.error(f"❌ Error creating employee: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@employee_bp.route('/<int:employee_id>', methods=['GET'])
def get_employee(employee_id: int):
    """
    Get specific employee with image details
    """
    try:
        if not employee_service or not face_service:
            return jsonify({'success': False, 'error': 'Services not initialized'}), 500
        
        # Get employee basic info
        employees = employee_service.get_all_employees()
        employee = next((emp for emp in employees if emp['id'] == employee_id), None)
        
        if not employee:
            return jsonify({'success': False, 'error': 'Employee not found'}), 404
        
        # Get employee images
        images = face_service.get_employee_images(employee_id)
        
        employee_data = {
            'id': employee['id'],
            'name': employee['name'],
            'employee_code': employee['employee_code'],
            'department': employee.get('department'),
            'position': employee.get('position'),
            'email': employee.get('email'),
            'phone': employee.get('phone'),
            'status': employee.get('status', 'active'),
            'created_at': employee.get('created_at'),
            'image_count': len(images),
            'images': images
        }
        
        return jsonify({
            'success': True,
            'data': employee_data
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting employee {employee_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@employee_bp.route('/<int:employee_id>', methods=['DELETE'])
def delete_employee(employee_id: int):
    """
    Delete an employee and all associated face data
    """
    try:
        if not employee_service or not face_service:
            return jsonify({'success': False, 'error': 'Services not initialized'}), 500
        
        # Get employee info before deletion
        employees = employee_service.get_all_employees()
        employee = next((emp for emp in employees if emp['id'] == employee_id), None)
        
        if not employee:
            return jsonify({'success': False, 'error': 'Employee not found'}), 404
        
        employee_name = employee['name']
        
        # Delete all employee images and vectors first
        images = face_service.get_employee_images(employee_id)
        for image in images:
            face_service.delete_employee_image(image['id'])
        
        # Delete employee (this should be implemented in employee_service)
        # For now, we'll mark as deleted in the enhanced service
        success = employee_service.delete_employee(employee_id) if hasattr(employee_service, 'delete_employee') else True
        
        if success:
            return jsonify({
                'success': True,
                'data': {
                    'message': f'Employee {employee_name} and all associated data deleted successfully',
                    'deleted_images': len(images)
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to delete employee'}), 500
        
    except Exception as e:
        logger.error(f"❌ Error deleting employee {employee_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@employee_bp.route('/<int:employee_id>/upload', methods=['POST'])
def upload_employee_image(employee_id: int):
    """
    Upload and process a face image for an employee with DeepFace vector extraction
    """
    try:
        if not employee_service or not face_service:
            return jsonify({'success': False, 'error': 'Required services not initialized'}), 500
        
        # Validate employee exists
        employees = employee_service.get_all_employees()
        employee = next((emp for emp in employees if emp['id'] == employee_id), None)
        
        if not employee:
            return jsonify({'success': False, 'error': 'Employee not found'}), 404
        
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        filename = file.filename or ''
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        if file_extension not in allowed_extensions:
            return jsonify({
                'success': False, 
                'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'
            }), 400
        
        # Process image with DeepFace vector extraction
        result = face_service.save_employee_image_with_vector(file, employee_id)
        
        if result['success']:
            return jsonify({
                'success': True,
                'data': {
                    'vector_id': result['vector_id'],
                    'image_path': result['image_path'],
                    'employee_name': result['employee_name'],
                    'vector_dimension': result['vector_dimension'],
                    'confidence': result['confidence'],
                    'total_images': result['total_images'],
                    'message': f'Image uploaded and processed successfully for {result["employee_name"]}'
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error occurred')
            }), 400
        
    except Exception as e:
        logger.error(f"❌ Error uploading image for employee {employee_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@employee_bp.route('/<int:employee_id>/images', methods=['GET'])
def get_employee_images(employee_id: int):
    """
    Get all images for a specific employee
    """
    try:
        if not face_service:
            return jsonify({'success': False, 'error': 'Face service not initialized'}), 500
        
        # Validate employee exists
        if employee_service:
            employees = employee_service.get_all_employees()
            employee = next((emp for emp in employees if emp['id'] == employee_id), None)
            
            if not employee:
                return jsonify({'success': False, 'error': 'Employee not found'}), 404
        
        # Get employee images
        images = face_service.get_employee_images(employee_id)
        
        return jsonify({
            'success': True,
            'data': images,
            'count': len(images),
            'employee_id': employee_id
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting images for employee {employee_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@employee_bp.route('/<int:employee_id>/vectors', methods=['GET'])
def get_employee_vectors(employee_id: int):
    """
    Get all face vectors for a specific employee (alias for images)
    """
    return get_employee_images(employee_id)

@employee_bp.route('/vectors/<int:vector_id>', methods=['DELETE'])
def delete_face_vector(vector_id: int):
    """
    Delete a specific face vector/image
    """
    try:
        if not face_service:
            return jsonify({'success': False, 'error': 'Face service not initialized'}), 500
        
        success = face_service.delete_employee_image(vector_id)
        
        if success:
            return jsonify({
                'success': True,
                'data': {
                    'message': 'Face vector and image deleted successfully',
                    'vector_id': vector_id
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Face vector not found'}), 404
        
    except Exception as e:
        logger.error(f"❌ Error deleting face vector {vector_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@employee_bp.route('/recognition/stats', methods=['GET'])
def get_recognition_stats():
    """
    Get face recognition performance statistics
    """
    try:
        if not face_service:
            return jsonify({'success': False, 'error': 'Face service not initialized'}), 500
        
        stats = face_service.get_recognition_stats()
        
        return jsonify({
            'success': True,
            'data': stats,
            'timestamp': current_app._get_current_object().config.get('CURRENT_TIME', 'N/A')
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting recognition stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@employee_bp.route('/recognition/threshold', methods=['POST'])
def update_recognition_threshold():
    """
    Update face recognition similarity threshold
    """
    try:
        if not face_service:
            return jsonify({'success': False, 'error': 'Face service not initialized'}), 500
        
        data = request.get_json()
        if not data or 'threshold' not in data:
            return jsonify({'success': False, 'error': 'Threshold value required'}), 400
        
        new_threshold = data['threshold']
        
        if not isinstance(new_threshold, (int, float)) or not (0.0 <= new_threshold <= 1.0):
            return jsonify({
                'success': False, 
                'error': 'Threshold must be a number between 0.0 and 1.0'
            }), 400
        
        face_service.update_threshold(float(new_threshold))
        
        return jsonify({
            'success': True,
            'data': {
                'threshold': float(new_threshold),
                'message': f'Recognition threshold updated to {new_threshold}'
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error updating threshold: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@employee_bp.route('/test/recognition', methods=['POST'])
def test_face_recognition():
    """
    Test face recognition with uploaded image
    """
    try:
        if not face_service:
            return jsonify({'success': False, 'error': 'Face service not initialized'}), 500
        
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Save temporary file
        import tempfile
        import cv2
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            
            # Read image with OpenCV
            frame = cv2.imread(tmp_file.name)
            
            if frame is None:
                os.unlink(tmp_file.name)
                return jsonify({'success': False, 'error': 'Invalid image file'}), 400
            
            # Perform face recognition
            results = face_service.recognize_faces_in_frame(frame)
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            return jsonify({
                'success': True,
                'data': {
                    'faces_detected': len(results),
                    'results': results,
                    'test_image_processed': True
                }
            })
        
    except Exception as e:
        logger.error(f"❌ Error in face recognition test: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@employee_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Employee endpoint not found'}), 404

@employee_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal employee service error'}), 500

@employee_bp.errorhandler(413)
def too_large(error):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB'}), 413