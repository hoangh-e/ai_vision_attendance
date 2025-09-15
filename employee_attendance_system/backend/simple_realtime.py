#!/usr/bin/env python3
"""
Simple Real-time Video Server - No complex dependencies
Works immediately for testing real-time video streaming
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import time
import random
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'simple_realtime_2025'

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False)

# Mock employee database
EMPLOYEES = [
    {'id': 1, 'name': 'Nguyễn Văn A', 'employee_id': 'EMP001', 'department': 'IT'},
    {'id': 2, 'name': 'Trần Thị B', 'employee_id': 'EMP002', 'department': 'HR'},
    {'id': 3, 'name': 'Lê Văn C', 'employee_id': 'EMP003', 'department': 'Finance'}
]

def simple_face_detection(frame):
    """
    Simple face detection using OpenCV - works immediately
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try to load face cascade
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            # Fallback - create mock detection
            return create_mock_detection(frame)
        
        if face_cascade.empty():
            return create_mock_detection(frame)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        results = []
        for (x, y, w, h) in faces:
            # Random recognition result for demo
            confidence = random.uniform(0.4, 0.95)
            
            if confidence > 0.6:
                employee = random.choice(EMPLOYEES)
                result = {
                    'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'employee': employee,
                    'confidence': float(confidence),
                    'type': 'known'
                }
            else:
                result = {
                    'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'employee': None,
                    'confidence': float(confidence),
                    'type': 'stranger'
                }
            
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Detection error: {e}")
        return create_mock_detection(frame)

def create_mock_detection(frame):
    """Create mock face detection for demo purposes"""
    h, w = frame.shape[:2]
    
    # Random chance of detecting a face
    if random.random() > 0.7:  # 30% chance
        # Create fake bounding box
        face_w = random.randint(80, 150)
        face_h = random.randint(80, 150)
        x = random.randint(0, max(1, w - face_w))
        y = random.randint(0, max(1, h - face_h))
        
        confidence = random.uniform(0.4, 0.95)
        
        if confidence > 0.6:
            employee = random.choice(EMPLOYEES)
            return [{
                'bbox': {'x': x, 'y': y, 'width': face_w, 'height': face_h},
                'employee': employee,
                'confidence': confidence,
                'type': 'known'
            }]
        else:
            return [{
                'bbox': {'x': x, 'y': y, 'width': face_w, 'height': face_h},
                'employee': None,
                'confidence': confidence,
                'type': 'stranger'
            }]
    
    return []

# Routes
@app.route('/')
def index():
    return """
    <h1>🎥 Simple Real-time Video Server</h1>
    <p>Ready for mobile video streaming!</p>
    <ul>
        <li><a href="/mobile">📱 Mobile Interface</a></li>
        <li><a href="/dashboard">💻 Dashboard</a></li>
    </ul>
    """

@app.route('/mobile')
def mobile():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📱 Mobile Video Stream</title>
    <style>
        body { margin: 0; padding: 20px; background: #1a1a1a; color: white; font-family: Arial; }
        .container { max-width: 400px; margin: 0 auto; text-align: center; }
        video { width: 100%; border-radius: 10px; margin: 20px 0; }
        canvas { position: absolute; top: 0; left: 0; pointer-events: none; }
        .video-container { position: relative; display: inline-block; }
        button { padding: 15px 30px; margin: 10px; border: none; border-radius: 25px; font-size: 16px; cursor: pointer; }
        .btn-start { background: #2ecc71; color: white; }
        .btn-stop { background: #e74c3c; color: white; }
        .stats { background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 20px 0; }
        .detection { background: rgba(46,204,113,0.2); border: 1px solid #2ecc71; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .stranger { background: rgba(231,76,60,0.2); border: 1px solid #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📱 Real-time Video</h1>
        <p id="status">🔴 Disconnected</p>
        
        <div class="video-container">
            <video id="video" autoplay muted playsinline></video>
            <canvas id="overlay"></canvas>
        </div>
        
        <button id="startBtn" class="btn-start">📹 Start Camera</button>
        <button id="stopBtn" class="btn-stop" style="display:none;">⏹ Stop</button>
        
        <div class="stats">
            <div>Frames: <span id="frames">0</span></div>
            <div>Faces: <span id="faces">0</span></div>
            <div>FPS: <span id="fps">0</span></div>
        </div>
        
        <div id="results"></div>
    </div>

    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script>
        let socket = io();
        let video = document.getElementById('video');
        let canvas = document.getElementById('overlay');
        let ctx = canvas.getContext('2d');
        let stream;
        let isStreaming = false;
        let frameCount = 0;
        let faceCount = 0;
        let lastTime = 0;

        socket.on('connect', () => {
            document.getElementById('status').innerHTML = '🟢 Connected';
        });

        socket.on('disconnect', () => {
            document.getElementById('status').innerHTML = '🔴 Disconnected';
        });

        socket.on('detection_result', (data) => {
            if (data.success) {
                drawResults(data.faces || []);
                updateStats(data.faces || []);
                showDetections(data.faces || []);
            }
        });

        document.getElementById('startBtn').onclick = async () => {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480, frameRate: 15 }
                });
                
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                };
                
                isStreaming = true;
                document.getElementById('startBtn').style.display = 'none';
                document.getElementById('stopBtn').style.display = 'inline-block';
                
                startCapture();
            } catch (err) {
                alert('Camera error: ' + err.message);
            }
        };

        document.getElementById('stopBtn').onclick = () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            video.srcObject = null;
            isStreaming = false;
            document.getElementById('startBtn').style.display = 'inline-block';
            document.getElementById('stopBtn').style.display = 'none';
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        };

        function startCapture() {
            if (!isStreaming) return;
            
            const capture = () => {
                if (!isStreaming) return;
                
                const tempCanvas = document.createElement('canvas');
                const tempCtx = tempCanvas.getContext('2d');
                tempCanvas.width = video.videoWidth || 640;
                tempCanvas.height = video.videoHeight || 480;
                
                tempCtx.drawImage(video, 0, 0);
                const frame = tempCanvas.toDataURL('image/jpeg', 0.7);
                
                socket.emit('video_frame', {
                    frame: frame,
                    timestamp: Date.now()
                });
                
                frameCount++;
                setTimeout(capture, 200); // 5 FPS
            };
            
            capture();
        }

        function drawResults(faces) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const scaleX = canvas.width / video.videoWidth;
            const scaleY = canvas.height / video.videoHeight;
            
            faces.forEach(face => {
                const bbox = face.bbox;
                const x = bbox.x * scaleX;
                const y = bbox.y * scaleY;
                const w = bbox.width * scaleX;
                const h = bbox.height * scaleY;
                
                ctx.strokeStyle = face.type === 'known' ? '#2ecc71' : '#e74c3c';
                ctx.lineWidth = 3;
                ctx.strokeRect(x, y, w, h);
                
                const label = face.type === 'known' ? 
                    `${face.employee.name} (${Math.round(face.confidence * 100)}%)` :
                    `Người lạ (${Math.round(face.confidence * 100)}%)`;
                
                ctx.fillStyle = face.type === 'known' ? '#2ecc71' : '#e74c3c';
                ctx.fillRect(x, y - 25, ctx.measureText(label).width + 10, 20);
                ctx.fillStyle = 'white';
                ctx.font = '12px Arial';
                ctx.fillText(label, x + 5, y - 10);
            });
        }

        function updateStats(faces) {
            faceCount += faces.length;
            document.getElementById('frames').textContent = frameCount;
            document.getElementById('faces').textContent = faceCount;
            
            const now = Date.now();
            if (lastTime > 0) {
                const fps = Math.round(1000 / (now - lastTime));
                document.getElementById('fps').textContent = fps;
            }
            lastTime = now;
        }

        function showDetections(faces) {
            const results = document.getElementById('results');
            results.innerHTML = '';
            
            faces.forEach(face => {
                const div = document.createElement('div');
                div.className = face.type === 'known' ? 'detection' : 'detection stranger';
                
                if (face.type === 'known') {
                    div.innerHTML = `
                        <strong>👤 ${face.employee.name}</strong><br>
                        📝 ${face.employee.employee_id}<br>
                        🏢 ${face.employee.department}<br>
                        🎯 ${Math.round(face.confidence * 100)}%
                    `;
                } else {
                    div.innerHTML = `
                        <strong>⚠️ Người lạ</strong><br>
                        🎯 ${Math.round(face.confidence * 100)}%
                    `;
                }
                
                results.appendChild(div);
            });
        }
    </script>
</body>
</html>
    """

@app.route('/dashboard')
def dashboard():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>💻 Dashboard</title>
    <style>
        body { margin: 0; padding: 20px; background: #1a1a1a; color: white; font-family: Arial; }
        .stats { display: flex; gap: 20px; margin-bottom: 20px; }
        .stat-card { background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; flex: 1; text-align: center; }
        .activity { background: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px; height: 400px; overflow-y: auto; }
        .activity-item { background: rgba(255,255,255,0.1); padding: 10px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>💻 Real-time Dashboard</h1>
    
    <div class="stats">
        <div class="stat-card">
            <h3>Connected Clients</h3>
            <div id="clients">0</div>
        </div>
        <div class="stat-card">
            <h3>Total Frames</h3>
            <div id="totalFrames">0</div>
        </div>
        <div class="stat-card">
            <h3>Faces Detected</h3>
            <div id="totalFaces">0</div>
        </div>
    </div>
    
    <div class="activity">
        <h2>🔄 Activity Log</h2>
        <div id="activity">Waiting for connections...</div>
    </div>

    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script>
        let socket = io();
        let clients = 0;
        let totalFrames = 0;
        let totalFaces = 0;

        socket.on('frame_processed', (data) => {
            totalFrames++;
            totalFaces += data.faces_detected || 0;
            
            document.getElementById('totalFrames').textContent = totalFrames;
            document.getElementById('totalFaces').textContent = totalFaces;
            
            addActivity(`📱 Frame processed: ${data.faces_detected} faces detected`);
        });

        function addActivity(message) {
            const activity = document.getElementById('activity');
            const item = document.createElement('div');
            item.className = 'activity-item';
            item.innerHTML = `[${new Date().toLocaleTimeString()}] ${message}`;
            activity.insertBefore(item, activity.firstChild);
            
            // Keep only last 20 items
            while (activity.children.length > 20) {
                activity.removeChild(activity.lastChild);
            }
        }
    </script>
</body>
</html>
    """

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    print(f"📱 Client connected")

@socketio.on('disconnect') 
def handle_disconnect():
    print(f"📱 Client disconnected")

@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        frame_data = data.get('frame')
        if not frame_data:
            emit('detection_result', {'success': False, 'error': 'No frame data'})
            return
        
        # Decode frame
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
        
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            emit('detection_result', {'success': False, 'error': 'Could not decode frame'})
            return
        
        # Detect faces
        faces = simple_face_detection(frame)
        
        # Send results
        result = {
            'success': True,
            'faces': faces,
            'timestamp': time.time(),
            'client_timestamp': data.get('timestamp', time.time())
        }
        
        emit('detection_result', result)
        
        # Broadcast to dashboard
        socketio.emit('frame_processed', {
            'faces_detected': len(faces),
            'timestamp': time.time()
        })
        
    except Exception as e:
        print(f"Frame processing error: {e}")
        emit('detection_result', {'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🎥 Starting Simple Real-time Video Server...")
    print("📱 Mobile: http://localhost:5004/mobile")
    print("💻 Dashboard: http://localhost:5004/dashboard") 
    print()
    
    socketio.run(app, host='0.0.0.0', port=5004, debug=True)