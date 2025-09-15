#!/usr/bin/env python3
"""
Simple HTTP server for testing - No SocketIO
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

# Static files
@app.route('/')
def index():
    return """
    <h1>🚀 BHK Tech - AI Vision Attendance System</h1>
    <p>Server is running successfully!</p>
    <ul>
        <li><a href="/api/employees">Test API: Get Employees</a></li>
        <li><a href="/mobile/simple">Mobile Interface (Simple)</a></li>
        <li><a href="/static/mobile_app/simple.html">Direct Mobile App</a></li>
    </ul>
    <p><strong>Note:</strong> Camera requires HTTPS on mobile devices</p>
    """

@app.route('/api/employees')
def get_employees():
    """Get list of employees for testing"""
    return jsonify({
        'success': True,
        'employees': [
            {'id': 1, 'name': 'Nguyễn Văn A', 'employee_id': 'EMP001', 'department': 'IT'},
            {'id': 2, 'name': 'Trần Thị B', 'employee_id': 'EMP002', 'department': 'HR'},
            {'id': 3, 'name': 'Lê Văn C', 'employee_id': 'EMP003', 'department': 'Finance'}
        ]
    })

@app.route('/api/employees', methods=['POST'])
def add_employee():
    """Add new employee"""
    data = request.get_json()
    return jsonify({
        'success': True,
        'message': 'Employee added successfully',
        'employee': {
            'id': 4,
            'name': data.get('name', 'Unknown'),
            'employee_id': data.get('employee_id', 'NEW001'),
            'department': data.get('department', 'General')
        }
    })

@app.route('/api/process_face', methods=['POST'])
def process_face_api():
    """Process face recognition - Mock response"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Không có dữ liệu hình ảnh'}), 400
        
        # Mock successful recognition
        return jsonify({
            'success': True,
            'employee': {
                'name': 'Demo User',
                'employee_id': 'DEMO001',
                'department': 'Testing'
            },
            'confidence': 0.95,
            'timestamp': data.get('timestamp', 'N/A'),
            'message': 'Mock recognition - Camera working!'
        })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi server: {str(e)}'
        }), 500

@app.route('/mobile/simple')
def mobile_simple():
    """Serve mobile simple interface"""
    return send_from_directory('../static/mobile_app', 'simple.html')

# Health check
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'OK',
        'message': 'Server is running',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🚀 Starting Simple HTTP Server...")
    print("📱 Access at: http://localhost:5001")
    print("📱 Mobile: http://localhost:5001/mobile/simple")
    print("🔗 API Test: http://localhost:5001/api/employees")
    print()
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)