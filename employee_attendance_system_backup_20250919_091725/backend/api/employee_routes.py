#!/usr/bin/env python3# Định nghĩa các route cho nhân viên

"""
employee_routes.py - EMPLOYEE MANAGEMENT API ROUTES
API endpoints for employee CRUD operations and face image management
"""

from flask import Blueprint, request, jsonify
import logging
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

# Create blueprint
employee_bp = Blueprint('employee', __name__, url_prefix='/api/employees')

# Services will be injected by main_app.py
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
            return jsonify({'success': False, 'error': 'Employee service not initialized'})
        
        employees = employee_service.get_all_employees()
        
        result = []
        for emp in employees:
            employee_data = {
                'id': emp.id,
                'name': emp.name,
                'employee_code': emp.employee_code,
                'department': emp.department,
                'position': emp.position,
                'email': emp.email,
                'phone': emp.phone,
                'created_at': emp.created_at.isoformat() if emp.created_at else None,
                'image_count': 0
            }
            
            # Get face image count
            if face_service:
                employee_data['image_count'] = face_service.get_employee_image_count(emp.id)
            
            result.append(employee_data)
        
        return jsonify({
            'success': True,
            'data': result,
            'total': len(result)
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting employees: {e}")
        return jsonify({'success': False, 'error': str(e)})

@employee_bp.route('', methods=['POST'])
def create_employee():
    """
    Create a new employee
    """
    try:
        if not employee_service:
            return jsonify({'success': False, 'error': 'Employee service not initialized'})
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'employee_code']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'error': f'Field {field} is required'})
        
        # Check if employee code already exists
        existing = employee_service.get_employee_by_code(data['employee_code'])
        if existing:
            return jsonify({'success': False, 'error': 'Employee code already exists'})
        
        # Create employee
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
                'employee_code': employee.employee_code,
                'message': f'Employee {employee.name} created successfully'
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error creating employee: {e}")
        return jsonify({'success': False, 'error': str(e)})

@employee_bp.route('/<int:employee_id>/upload', methods=['POST'])
def upload_employee_image(employee_id: int):
    """
    Upload and process a face image for an employee
    """
    try:
        if not employee_service or not face_service:
            return jsonify({'success': False, 'error': 'Required services not initialized'})
        
        # Check if employee exists
        employee = employee_service.get_employee_by_id(employee_id)
        if not employee:
            return jsonify({'success': False, 'error': 'Employee not found'})
        
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Check current image count
        current_count = face_service.get_employee_image_count(employee_id)
        if current_count >= 10:
            return jsonify({'success': False, 'error': 'Employee already has maximum 10 images'})
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        filename = file.filename or ''
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        if file_extension not in allowed_extensions:
            return jsonify({'success': False, 'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'})
        
        # Save image and extract face vector
        vector_face = face_service.save_image_and_vector(file, employee_id)
        
        return jsonify({
            'success': True,
            'data': {
                'id': vector_face.id,
                'image_path': vector_face.image_path,
                'employee_id': employee_id,
                'current_count': current_count + 1,
                'message': f'Image uploaded successfully for {employee.name}'
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error uploading image for employee {employee_id}: {e}")
        return jsonify({'success': False, 'error': str(e)})

@employee_bp.route('/<int:employee_id>', methods=['DELETE'])
def delete_employee(employee_id: int):
    """
    Delete an employee and all associated face data
    """
    try:
        if not employee_service:
            return jsonify({'success': False, 'error': 'Employee service not initialized'})
        
        # Check if employee exists
        employee = employee_service.get_employee_by_id(employee_id)
        if not employee:
            return jsonify({'success': False, 'error': 'Employee not found'})
        
        employee_name = employee.name
        
        # Delete associated face vectors first
        if face_service:
            vectors = face_service.get_employee_vectors(employee_id)
            for vector in vectors:
                face_service.delete_face_vector(vector.id)
        
        # Delete employee
        success = employee_service.delete_employee(employee_id)
        
        if success:
            return jsonify({
                'success': True,
                'data': {
                    'message': f'Employee {employee_name} and all associated data deleted successfully'
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to delete employee'})
        
    except Exception as e:
        logger.error(f"❌ Error deleting employee {employee_id}: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Face vector management routes
@employee_bp.route('/vectors/<int:vector_id>', methods=['DELETE'])
def delete_face_vector(vector_id: int):
    """
    Delete a specific face vector/image
    """
    try:
        if not face_service:
            return jsonify({'success': False, 'error': 'Face service not initialized'})
        
        success = face_service.delete_face_vector(vector_id)
        
        if success:
            return jsonify({
                'success': True,
                'data': {
                    'message': 'Face vector deleted successfully'
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Face vector not found'})
        
    except Exception as e:
        logger.error(f"❌ Error deleting face vector {vector_id}: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Error handlers
@employee_bp.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Employee endpoint not found'}), 404

@employee_bp.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal employee service error'}), 500