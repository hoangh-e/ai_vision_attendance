//  MOBILE REALTIME VIDEO STREAMING - FIXED VERSION
// File: frontend/static/js/mobile-stream.js

class MobileVideoStreamer {
    constructor() {
        this.socket = null;
        this.videoElement = null;
        this.canvas = null;
        this.ctx = null;
        this.stream = null;
        this.isStreaming = false;
        this.frameRate = 10; // Reduced for mobile stability
        this.quality = 0.7; // Reduced for mobile performance
        this.intervalId = null;
        this.clientId = null;
        
        // Stats
        this.stats = {
            framesSent: 0,
            fps: 0,
            lastFpsUpdate: Date.now(),
            framesInLastSecond: 0
        };
    }
    
    async initialize() {
        try {
            console.log(' Initializing mobile video streamer...');
            
            // Initialize video elements FIRST
            this.videoElement = document.getElementById('videoElement');
            this.canvas = document.getElementById('frameCanvas');
            
            if (!this.videoElement || !this.canvas) {
                throw new Error('Video elements not found in DOM');
            }
            
            this.ctx = this.canvas.getContext('2d');
            
            // Initialize Socket.IO AFTER DOM is ready
            await this.initializeSocket();
            
            console.log(' Mobile video streamer initialized');
            
            // Show ready message
            this.showStatus(' Sẵn sàng! Nhấn "Bắt đầu Stream" để bắt đầu', 'success');
            
            return true;
            
        } catch (error) {
            console.error(' Failed to initialize mobile streamer:', error);
            this.showStatus(' Lỗi khởi tạo: ' + error.message, 'error');
            throw error;
        }
    }
    
    async initializeSocket() {
        return new Promise((resolve, reject) => {
            try {
                // Initialize Socket.IO with proper config for mobile
                this.socket = io(window.location.origin, {
                    transports: ['websocket', 'polling'],
                    timeout: 20000,
                    forceNew: true,
                    reconnection: true,
                    reconnectionDelay: 1000,
                    reconnectionAttempts: 5
                });
                
                this.setupSocketEvents();
                
                // Wait for connection
                this.socket.on('connect', () => {
                    console.log(' Socket connected successfully');
                    resolve();
                });
                
                this.socket.on('connect_error', (error) => {
                    console.error(' Socket connection error:', error);
                    reject(error);
                });
                
                // Timeout fallback
                setTimeout(() => {
                    if (!this.socket.connected) {
                        reject(new Error('Socket connection timeout'));
                    }
                }, 10000);
                
            } catch (error) {
                reject(error);
            }
        });
    }
    
    setupSocketEvents() {
        this.socket.on('connect', () => {
            console.log(' Connected to server');
            this.updateConnectionStatus(true);
            
            // Generate unique client ID
            this.clientId = 'mobile_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            
            // Register as mobile client
            this.socket.emit('register_mobile_client', {
                type: 'mobile_video_streamer',
                user_agent: navigator.userAgent,
                screen_size: `${screen.width}x${screen.height}`,
                timestamp: Date.now(),
                client_id: this.clientId
            });
        });
        
        this.socket.on('disconnect', () => {
            console.log(' Disconnected from server');
            this.updateConnectionStatus(false);
            this.stopStreaming();
        });
        
        this.socket.on('mobile_registered', (data) => {
            console.log(' Mobile client registered:', data);
            this.showStatus(' Đã kết nối tới server', 'success');
        });
        
        this.socket.on('detection_result', (data) => {
            this.displayDetectionResults(data);
        });
        
        this.socket.on('stats_update', (data) => {
            this.updateServerStats(data.stats);
        });
        
        this.socket.on('detection_status_changed', (data) => {
            console.log(' Detection toggled:', data.active);
        });
    }
    
    async setupCamera() {
        try {
            console.log('🎥 Setting up camera...');
            this.showStatus('🎥 Đang truy cập camera...', 'info');
            
            // ⭐ FIX: Kiểm tra HTTPS requirement
            if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
                throw new Error('Camera requires HTTPS connection. Please use https://');
            }
            
            // Stop existing stream if any
            if (this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            }
            
            // Check if getUserMedia is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Camera API không được hỗ trợ trên trình duyệt này');
            }
            
            // ⭐ FIX: Enhanced constraints with fallback
            const constraints = {
                video: {
                    facingMode: 'user', // Front camera
                    width: { ideal: 640, max: 1280 },
                    height: { ideal: 480, max: 720 },
                    frameRate: { ideal: 15, max: 30 }
                },
                audio: false
            };
            
            console.log('🎥 Requesting camera with constraints:', constraints);
            
            try {
                this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (firstError) {
                console.warn('⚠️ First camera request failed, trying fallback...', firstError);
                
                // ⭐ FIX: Fallback with minimal constraints
                const fallbackConstraints = {
                    video: true,
                    audio: false
                };
                
                this.stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
            }
            
            console.log('✅ Camera stream obtained');
            
            // Attach stream to video element
            this.videoElement.srcObject = this.stream;
            
            // ⭐ FIX: Wait for metadata to load
            return new Promise((resolve, reject) => {
                this.videoElement.addEventListener('loadedmetadata', () => {
                    this.canvas.width = this.videoElement.videoWidth || 640;
                    this.canvas.height = this.videoElement.videoHeight || 480;
                    console.log('✅ Camera setup successful');
                    this.showStatus('✅ Camera sẵn sàng!', 'success');
                    resolve();
                });
                
                this.videoElement.addEventListener('error', reject);
                
                // Start playback
                this.videoElement.play().catch(reject);
            });
            
        } catch (error) {
            console.error('❌ Camera setup failed:', error);
            
            // ⭐ FIX: Better error messages
            let errorMessage = 'Không thể truy cập camera';
            
            if (error.name === 'NotAllowedError') {
                errorMessage = 'Vui lòng cấp quyền truy cập camera';
            } else if (error.name === 'NotFoundError') {
                errorMessage = 'Không tìm thấy camera trên thiết bị';
            } else if (error.name === 'NotSupportedError') {
                errorMessage = 'Camera không được hỗ trợ';
            } else if (error.name === 'NotReadableError') {
                errorMessage = 'Camera đang được sử dụng bởi ứng dụng khác';
            } else if (error.message.includes('HTTPS')) {
                errorMessage = 'Cần HTTPS để truy cập camera. Vui lòng sử dụng https://';
            }
            
            this.showStatus('❌ ' + errorMessage, 'error');
            throw new Error(errorMessage);
        }
    }
    
    async startStreaming() {
        if (this.isStreaming) {
            console.log(' Already streaming');
            return;
        }
        
        try {
            // Setup camera first
            await this.setupCamera();
            
            // Start streaming
            this.isStreaming = true;
            const frameInterval = 1000 / this.frameRate; // ms per frame
            
            this.intervalId = setInterval(() => {
                this.captureAndSendFrame();
            }, frameInterval);
            
            console.log(`Started streaming at ${this.frameRate} FPS`);
            this.updateUI('streaming');
            this.showStatus(`Đang stream với ${this.frameRate} FPS`, 'success');
            
        } catch (error) {
            console.error(' Failed to start streaming:', error);
            this.showStatus(' Lỗi khởi động stream: ' + error.message, 'error');
            this.isStreaming = false;
        }
    }
    
    stopStreaming() {
        if (!this.isStreaming) return;
        
        this.isStreaming = false;
        
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        
        // Stop camera stream
        if (this.stream) {
            this.stream.getTracks().forEach(track => {
                track.stop();
                console.log(' Stopped camera track:', track.kind);
            });
            this.stream = null;
        }
        
        // Clear video element
        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }
        
        console.log(' Stopped streaming');
        this.updateUI('stopped');
        this.showStatus(' Đã dừng stream', 'info');
    }
    
    captureAndSendFrame() {
        if (!this.videoElement || !this.canvas || !this.socket || !this.socket.connected) {
            return;
        }
        
        try {
            // Check if video is playing
            if (this.videoElement.readyState < 2) {
                return; // Video not ready
            }
            
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
                client_id: this.clientId,
                resolution: `${this.canvas.width}x${this.canvas.height}`,
                quality: this.quality
            });
            
            // Update stats
            this.updateStats();
            
        } catch (error) {
            console.error(' Frame capture failed:', error);
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
            
            // Update detection info
            this.updateDetectionInfo(data);
            
            // Update detection count
            const detectionCount = document.getElementById('detectionCount');
            if (detectionCount) {
                detectionCount.textContent = data.faces.length;
            }
        } else {
            this.updateDetectionInfo({ faces: [] });
        }
    }
    
    drawBoundingBox(face, overlay) {
        const box = document.createElement('div');
        box.className = 'bounding-box';
        
        // Calculate position relative to video display size
        const videoRect = this.videoElement.getBoundingClientRect();
        const scaleX = videoRect.width / this.canvas.width;
        const scaleY = videoRect.height / this.canvas.height;
        
        const x = Math.max(0, face.x * scaleX);
        const y = Math.max(0, face.y * scaleY);
        const width = Math.min(videoRect.width - x, face.width * scaleX);
        const height = Math.min(videoRect.height - y, face.height * scaleY);
        
        box.style.left = `${x}px`;
        box.style.top = `${y}px`;
        box.style.width = `${width}px`;
        box.style.height = `${height}px`;
        
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
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (box.parentNode) {
                box.parentNode.removeChild(box);
            }
        }, 3000);
    }
    
    async switchCamera() {
        try {
            this.showStatus(' Đang đổi camera...', 'info');
            
            // Get current facing mode
            const currentTrack = this.stream.getVideoTracks()[0];
            const settings = currentTrack.getSettings();
            const newFacingMode = settings.facingMode === 'user' ? 'environment' : 'user';
            
            // Stop current stream
            this.stream.getTracks().forEach(track => track.stop());
            
            // Get new stream
            const constraints = {
                video: {
                    facingMode: newFacingMode,
                    width: { ideal: 640, max: 1280 },
                    height: { ideal: 480, max: 720 },
                    frameRate: { ideal: 15, max: 30 }
                },
                audio: false
            };
            
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Update video element
            this.videoElement.srcObject = this.stream;
            await this.videoElement.play();
            
            console.log(`Switched to ${newFacingMode} camera`);
            this.showStatus(`Đã chuyển sang camera ${newFacingMode === 'user' ? 'trước' : 'sau'}`, 'success');
            
        } catch (error) {
            console.error(' Camera switch failed:', error);
            this.showStatus(' Không thể chuyển camera', 'error');
        }
    }
    
    updateStats() {
        this.stats.framesSent++;
        this.stats.framesInLastSecond++;
        
        const now = Date.now();
        if (now - this.stats.lastFpsUpdate >= 1000) {
            this.stats.fps = this.stats.framesInLastSecond;
            this.stats.framesInLastSecond = 0;
            this.stats.lastFpsUpdate = now;
            
            // Update UI
            const fpsElement = document.getElementById('fpsDisplay');
            const framesElement = document.getElementById('framesCount');
            
            if (fpsElement) {
                fpsElement.textContent = `${this.stats.fps} FPS`;
            }
            if (framesElement) {
                framesElement.textContent = this.stats.framesSent;
            }
        }
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            statusElement.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
            statusElement.textContent = connected ? ' Đã kết nối' : ' Mất kết nối';
        }
    }
    
    updateDetectionInfo(data) {
        const infoElement = document.getElementById('detectionInfo');
        if (!infoElement) return;
        
        if (data.faces && data.faces.length > 0) {
            const knownFaces = data.faces.filter(f => f.employee).length;
            const unknownFaces = data.faces.length - knownFaces;
            
            infoElement.innerHTML = `
                <div class="detection-result">
                    ${data.faces.length} khuôn mặt | 
                    ${knownFaces} nhận diện | 
                    ${unknownFaces} người lạ
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
            if (startBtn) {
                startBtn.style.display = 'none';
                startBtn.disabled = true;
            }
            if (stopBtn) {
                stopBtn.style.display = 'inline-block';
                stopBtn.disabled = false;
            }
        } else {
            if (startBtn) {
                startBtn.style.display = 'inline-block';
                startBtn.disabled = false;
            }
            if (stopBtn) {
                stopBtn.style.display = 'none';
                stopBtn.disabled = true;
            }
        }
    }
    
    updateServerStats(data) {
        const statsElement = document.getElementById('serverStats');
        if (statsElement && data) {
            statsElement.innerHTML = `
                <div class="server-stats">
                    Server: ${data.processing?.current_fps || 0} FPS |
                    Clients: ${data.clients?.total_clients || 0} |
                    Detection: ${data.processing?.detection_active ? 'ON' : 'OFF'}
                </div>
            `;
        }
    }
    
    showStatus(message, type = 'info') {
        const statusDiv = document.getElementById('errorMessage');
        if (statusDiv) {
            statusDiv.textContent = message;
            statusDiv.style.display = 'block';
            
            // Set color based on type
            statusDiv.style.background = type === 'error' ? 'rgba(239, 68, 68, 0.2)' :
                                       type === 'success' ? 'rgba(34, 197, 94, 0.2)' :
                                       'rgba(59, 130, 246, 0.2)';
                                       
            statusDiv.style.borderColor = type === 'error' ? '#ef4444' :
                                         type === 'success' ? '#22c55e' :
                                         '#3b82f6';
            
            // Auto-hide after 5 seconds for non-error messages
            if (type !== 'error') {
                setTimeout(() => {
                    statusDiv.style.display = 'none';
                }, 5000);
            }
        }
    }
}

// Initialize when page loads
let mobileStreamer;

document.addEventListener('DOMContentLoaded', async () => {
    try {
        console.log(' DOM loaded, initializing mobile app...');
        
        mobileStreamer = new MobileVideoStreamer();
        await mobileStreamer.initialize();
        
        // Setup button events
        const startBtn = document.getElementById('startStreamBtn');
        const stopBtn = document.getElementById('stopStreamBtn');
        const switchBtn = document.getElementById('switchCameraBtn');
        const qualitySlider = document.getElementById('qualitySlider');
        const qualityValue = document.getElementById('qualityValue');
        
        if (startBtn) {
            startBtn.addEventListener('click', async () => {
                console.log('🎥 Start button clicked');
                startBtn.disabled = true;
                try {
                    await mobileStreamer.startStreaming();
                } catch (error) {
                    console.error('❌ Start streaming error:', error);
                } finally {
                    startBtn.disabled = false;
                }
            });
        }
        
        if (stopBtn) {
            stopBtn.addEventListener('click', () => {
                console.log('⏹️ Stop button clicked');
                mobileStreamer.stopStreaming();
            });
        }
        
        if (switchBtn) {
            switchBtn.addEventListener('click', async () => {
                console.log('🔄 Switch camera button clicked');
                switchBtn.disabled = true;
                try {
                    await mobileStreamer.switchCamera();
                } catch (error) {
                    console.error('❌ Switch camera error:', error);
                } finally {
                    setTimeout(() => {
                        switchBtn.disabled = false;
                    }, 1000);
                }
            });
        }
        
        if (qualitySlider && qualityValue) {
            qualitySlider.addEventListener('input', (e) => {
                const value = Math.round(e.target.value * 100);
                qualityValue.textContent = `${value}%`;
                
                if (mobileStreamer) {
                    mobileStreamer.quality = parseFloat(e.target.value);
                    console.log(`🎛️ Quality changed to: ${value}%`);
                }
            });
        }
        
        console.log(' Mobile app setup complete');
        
        // Store globally for debugging
        window.mobileStreamer = mobileStreamer;
        
    } catch (error) {
        console.error(' Failed to initialize mobile app:', error);
        
        const errorElement = document.getElementById('errorMessage');
        if (errorElement) {
            errorElement.innerHTML = `<strong>❌ Lỗi khởi tạo ứng dụng:</strong><br>${error.message}<br><br><strong>💡 Khắc phục:</strong><br>• Đảm bảo truy cập qua HTTPS<br>• Cấp quyền camera cho trình duyệt<br>• Kiểm tra kết nối mạng<br>• Thử làm mới trang (F5)`;
            errorElement.style.display = 'block';
        }
    }
});

// Export for global access
window.MobileVideoStreamer = MobileVideoStreamer;
