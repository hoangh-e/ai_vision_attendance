#!/usr/bin/env python3
"""
Simple HTTPS server for mobile camera access - No SocketIO complexity
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
    <h1>🔒 BHK Tech - HTTPS AI Vision Attendance</h1>
    <p>HTTPS Server is running for mobile camera access!</p>
    <ul>
        <li><a href="/api/employees">Test API: Get Employees</a></li>
        <li><a href="/mobile/simple">📱 Mobile Interface (Camera Ready)</a></li>
        <li><a href="/health">Health Check</a></li>
    </ul>
    <p><strong>✅ HTTPS Enabled:</strong> Mobile camera should work!</p>
    <p><strong>📱 Mobile URL:</strong> https://192.168.10.119:5002/mobile/simple</p>
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

@app.route('/api/process_face', methods=['POST'])
def process_face_api():
    """Process face recognition - Mock response for testing"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Không có dữ liệu hình ảnh'}), 400
        
        # Mock successful recognition
        import time
        return jsonify({
            'success': True,
            'employee': {
                'name': 'HTTPS Test User',
                'employee_id': 'HTTPS001',
                'department': 'Mobile Testing'
            },
            'confidence': 0.98,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'message': '🔒 HTTPS Camera Recognition Working!'
        })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Lỗi server: {str(e)}'
        }), 500

@app.route('/mobile/simple')
def mobile_simple():
    """Serve mobile simple interface"""
    try:
        return send_from_directory('../static/mobile_app', 'simple.html')
    except FileNotFoundError:
        return """
        <h1>📱 Mobile Camera Test</h1>
        <p>Mobile interface file not found. Testing camera access:</p>
        <button onclick="testCamera()">Test Camera</button>
        <video id="video" width="300" height="200" autoplay muted playsinline></video>
        <script>
        async function testCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({video: true});
                document.getElementById('video').srcObject = stream;
                alert('✅ Camera access successful with HTTPS!');
            } catch (err) {
                alert('❌ Camera error: ' + err.message);
            }
        }
        </script>
        """

# Health check
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'OK',
        'message': '🔒 HTTPS Server is running',
        'ssl': 'enabled',
        'camera_ready': True,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🔒 Starting Simple HTTPS Server...")
    print("📱 HTTPS Access: https://localhost:5002")
    print("📱 Mobile Camera: https://192.168.10.119:5002/mobile/simple")
    print("🔗 API Test: https://localhost:5002/api/employees")
    print()
    print("⚠️  You'll see security warnings - click 'Advanced' -> 'Proceed'")
    print("✅ Camera will work on mobile with HTTPS!")
    print()
    
    try:
        app.run(host='0.0.0.0', 
                port=5002, 
                debug=True, 
                ssl_context='adhoc',  # Auto-generate SSL certificate
                threaded=True)
    except Exception as e:
        print(f"❌ HTTPS failed: {e}")
        print("Falling back to HTTP...")
        app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)