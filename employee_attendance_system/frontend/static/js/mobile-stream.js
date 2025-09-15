// 📱 MOBILE REALTIME VIDEO STREAMING
// File: frontend/static/js/mobile-stream.js

class MobileVideoStreamer {
    constructor() {
        this.socket = null;
        this.videoElement = null;
        this.canvas = null;
        this.ctx = null;
        this.stream = null;
        this.isStreaming = false;
        this.frameRate = 15; // FPS
        this.quality = 0.8; // JPEG quality
        this.intervalId = null;
        
        // Stats
        this.stats = {
            framesSent: 0,
            fps: 0,
            lastFpsUpdate: Date.now()
        };
    }
    
    async initialize() {
        try {
            // Initialize Socket.IO
            this.socket = io(window.location.origin, {
                transports: ['websocket', 'polling'],
                timeout: 10000
            });
            
            this.setupSocketEvents();
            
            // Initialize video elements
            this.videoElement = document.getElementById('videoElement');
            this.canvas = document.getElementById('frameCanvas');
            this.ctx = this.canvas.getContext('2d');
            
            // Setup camera
            await this.setupCamera();
            
            console.log('✅ Mobile video streamer initialized');
            return true;
            
        } catch (error) {
            console.error('❌ Failed to initialize mobile streamer:', error);
            throw error;
        }
    }
    
    setupSocketEvents() {
        this.socket.on('connect', () => {
            console.log('✅ Connected to server');
            this.updateConnectionStatus(true);
            
            // Register as mobile client
            this.socket.emit('register_mobile_client', {
                type: 'mobile_video_streamer',
                user_agent: navigator.userAgent,
                screen_size: `${screen.width}x${screen.height}`,
                timestamp: Date.now()
            });
        });
        
        this.socket.on('disconnect', () => {
            console.log('❌ Disconnected from server');
            this.updateConnectionStatus(false);
            this.stopStreaming();
        });
        
        this.socket.on('detection_result', (data) => {
            this.displayDetectionResults(data);
        });
        
        this.socket.on('server_stats', (data) => {
            this.updateServerStats(data);
        });
    }
    
    async setupCamera() {
        try {
            // Request camera access
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user', // Front camera
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: this.frameRate }
                },
                audio: false
            });
            
            // Attach stream to video element
            this.videoElement.srcObject = this.stream;
            
            // Setup canvas size when video loads
            this.videoElement.addEventListener('loadedmetadata', () => {
                this.canvas.width = this.videoElement.videoWidth;
                this.canvas.height = this.videoElement.videoHeight;
                console.log(`📹 Camera setup: ${this.canvas.width}x${this.canvas.height}`);
            });
            
            await this.videoElement.play();
            
        } catch (error) {
            console.error('❌ Camera setup failed:', error);
            throw new Error('Không thể truy cập camera. Vui lòng cấp quyền camera.');
        }
    }
    
    startStreaming() {
        if (this.isStreaming) return;
        
        this.isStreaming = true;
        const frameInterval = 1000 / this.frameRate; // ms per frame
        
        this.intervalId = setInterval(() => {
            this.captureAndSendFrame();
        }, frameInterval);
        
        console.log(`🎥 Started streaming at ${this.frameRate} FPS`);
        this.updateUI('streaming');
    }
    
    stopStreaming() {
        if (!this.isStreaming) return;
        
        this.isStreaming = false;
        
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        
        console.log('⏹️ Stopped streaming');
        this.updateUI('stopped');
    }
    
    captureAndSendFrame() {
        if (!this.videoElement || !this.canvas || !this.socket) return;
        
        try {
            // Draw current video frame to canvas
            this.ctx.drawImage(
                this.videoElement, 
                0, 0, 
                this.canvas.width, 
                this.canvas.height
            );
            
            // Convert to base64 JPEG
            const frameData = this.canvas.toDataURL('image/jpeg', this.quality);
            
            // Send frame to server
            this.socket.emit('video_frame', {
                frame: frameData,
                timestamp: Date.now(),
                client_id: this.socket.id,
                resolution: `${this.canvas.width}x${this.canvas.height}`,
                quality: this.quality
            });
            
            // Update stats
            this.updateStats();
            
        } catch (error) {
            console.error('❌ Frame capture failed:', error);
        }
    }
    
    displayDetectionResults(data) {
        const overlay = document.getElementById('detectionOverlay');
        if (!overlay) return;
        
        // Clear previous results
        overlay.innerHTML = '';
        
        if (data.success && data.faces && data.faces.length > 0) {
            data.faces.forEach(face => {
                this.drawBoundingBox(face, overlay);
            });
        }
        
        // Update detection info
        this.updateDetectionInfo(data);
    }
    
    drawBoundingBox(face, overlay) {
        const box = document.createElement('div');
        box.className = 'bounding-box';
        
        // Calculate position relative to video
        const videoRect = this.videoElement.getBoundingClientRect();
        const scaleX = videoRect.width / this.canvas.width;
        const scaleY = videoRect.height / this.canvas.height;
        
        box.style.left = `${face.x * scaleX}px`;
        box.style.top = `${face.y * scaleY}px`;
        box.style.width = `${face.width * scaleX}px`;
        box.style.height = `${face.height * scaleY}px`;
        
        // Add label
        const label = document.createElement('div');
        label.className = 'face-label';
        
        if (face.employee) {
            label.textContent = `${face.employee.name} (${Math.round(face.confidence * 100)}%)`;
            box.classList.add('known-face');
        } else {
            label.textContent = 'Người lạ';
            box.classList.add('unknown-face');
        }
        
        box.appendChild(label);
        overlay.appendChild(box);
    }
    
    async switchCamera() {
        try {
            // Get current facing mode
            const currentTrack = this.stream.getVideoTracks()[0];
            const settings = currentTrack.getSettings();
            const newFacingMode = settings.facingMode === 'user' ? 'environment' : 'user';
            
            // Stop current stream
            this.stream.getTracks().forEach(track => track.stop());
            
            // Get new stream
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: newFacingMode,
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: this.frameRate }
                },
                audio: false
            });
            
            // Update video element
            this.videoElement.srcObject = this.stream;
            await this.videoElement.play();
            
            console.log(`📱 Switched to ${newFacingMode} camera`);
            
        } catch (error) {
            console.error('❌ Camera switch failed:', error);
            alert('Không thể chuyển camera');
        }
    }
    
    updateStats() {
        this.stats.framesSent++;
        
        const now = Date.now();
        if (now - this.stats.lastFpsUpdate >= 1000) {
            this.stats.fps = this.stats.framesSent;
            this.stats.framesSent = 0;
            this.stats.lastFpsUpdate = now;
            
            // Update UI
            const fpsElement = document.getElementById('fpsDisplay');
            if (fpsElement) {
                fpsElement.textContent = `${this.stats.fps} FPS`;
            }
        }
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            statusElement.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
            statusElement.textContent = connected ? '🟢 Đã kết nối' : '🔴 Mất kết nối';
        }
    }
    
    updateDetectionInfo(data) {
        const infoElement = document.getElementById('detectionInfo');
        if (!infoElement) return;
        
        if (data.success && data.faces && data.faces.length > 0) {
            const knownFaces = data.faces.filter(f => f.employee).length;
            const unknownFaces = data.faces.length - knownFaces;
            
            infoElement.innerHTML = `
                <div class="detection-result">
                    👥 ${data.faces.length} khuôn mặt | 
                    ✅ ${knownFaces} nhận diện | 
                    ❓ ${unknownFaces} người lạ
                </div>
            `;
        } else {
            infoElement.innerHTML = '<div class="no-detection">Không phát hiện khuôn mặt</div>';
        }
    }
    
    updateUI(state) {
        const startBtn = document.getElementById('startStreamBtn');
        const stopBtn = document.getElementById('stopStreamBtn');
        
        if (state === 'streaming') {
            if (startBtn) startBtn.style.display = 'none';
            if (stopBtn) stopBtn.style.display = 'inline-block';
        } else {
            if (startBtn) startBtn.style.display = 'inline-block';
            if (stopBtn) stopBtn.style.display = 'none';
        }
    }
    
    updateServerStats(data) {
        // Update server statistics display
        const statsElement = document.getElementById('serverStats');
        if (statsElement && data) {
            statsElement.innerHTML = `
                <div class="server-stats">
                    📊 Server: ${data.processing?.current_fps || 0} FPS |
                    👥 Clients: ${data.clients?.total_clients || 0} |
                    🔍 Detection: ${data.processing?.detection_active ? 'ON' : 'OFF'}
                </div>
            `;
        }
    }
}

// Initialize when page loads
let mobileStreamer;

document.addEventListener('DOMContentLoaded', async () => {
    try {
        mobileStreamer = new MobileVideoStreamer();
        await mobileStreamer.initialize();
        
        // Setup button events
        const startBtn = document.getElementById('startStreamBtn');
        const stopBtn = document.getElementById('stopStreamBtn');
        const switchBtn = document.getElementById('switchCameraBtn');
        const toggleDetectionBtn = document.getElementById('toggleDetectionBtn');
        
        if (startBtn) {
            startBtn.addEventListener('click', () => {
                mobileStreamer.startStreaming();
            });
        }
        
        if (stopBtn) {
            stopBtn.addEventListener('click', () => {
                mobileStreamer.stopStreaming();
            });
        }
        
        if (switchBtn) {
            switchBtn.addEventListener('click', () => {
                mobileStreamer.switchCamera();
            });
        }
        
        if (toggleDetectionBtn) {
            toggleDetectionBtn.addEventListener('click', () => {
                // Toggle detection on server
                const currentState = mobileStreamer.socket && mobileStreamer.socket.connected;
                if (currentState) {
                    mobileStreamer.socket.emit('toggle_detection', {
                        active: !document.getElementById('toggleDetectionBtn').classList.contains('active')
                    });
                }
            });
        }
        
    } catch (error) {
        console.error('❌ Failed to initialize mobile app:', error);
        
        // Show error message to user
        const errorElement = document.getElementById('errorMessage');
        if (errorElement) {
            errorElement.textContent = 'Lỗi khởi tạo ứng dụng: ' + error.message;
            errorElement.style.display = 'block';
        } else {
            alert('Lỗi khởi tạo ứng dụng: ' + error.message);
        }
    }
});

// Export for global access
window.MobileVideoStreamer = MobileVideoStreamer;