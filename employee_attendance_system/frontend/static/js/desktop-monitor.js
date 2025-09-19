/**
 * desktop-monitor.js - DESKTOP MONITORING INTERFACE
 * Handles desktop video monitoring, employee management, and system controls
 */

class DesktopMonitor {
    constructor() {
        this.socket = io();
        this.currentFrame = null;
        this.detectionActive = false;
        this.employees = [];
        this.stats = {
            totalFrames: 0,
            totalDetections: 0,
            mobileClients: 0,
            desktopClients: 0
        };
        
        this.init();
    }
    
    init() {
        this.setupSocketEvents();
        this.setupUIEvents();
        this.loadEmployees();
        this.startStatsUpdate();
        
        console.log('🖥️ Desktop Monitor initialized');
    }
    
    setupSocketEvents() {
        // Connection events
        this.socket.on('connect', () => {
            console.log('✅ Connected to server');
            this.updateConnectionStatus(true);
            this.registerAsDesktop();
        });
        
        this.socket.on('disconnect', () => {
            console.log('❌ Disconnected from server');
            this.updateConnectionStatus(false);
        });
        
        // Registration response
        this.socket.on('desktop_registered', (data) => {
            console.log('🖥️ Desktop client registered:', data);
            this.detectionActive = data.detection_active;
            this.updateDetectionToggle();
        });
        
        // Video frame updates
        this.socket.on('video_frame_update', (data) => {
            this.handleVideoFrame(data);
        });
        
        // Detection state changes
        this.socket.on('detection_state_changed', (data) => {
            this.detectionActive = data.active;
            this.updateDetectionToggle();
            this.showNotification(
                `Detection ${data.active ? 'activated' : 'deactivated'}`,
                data.active ? 'success' : 'info'
            );
        });
        
        // Client count updates
        this.socket.on('client_count_update', (data) => {
            this.stats.mobileClients = data.mobile_count;
            this.stats.desktopClients = data.desktop_count;
            this.updateStatsDisplay();
        });
        
        // Mobile client events
        this.socket.on('mobile_connected', (data) => {
            this.stats.mobileClients = data.mobile_count;
            this.showNotification('📱 Mobile client connected', 'info');
            this.updateStatsDisplay();
        });
    }
    
    setupUIEvents() {
        // Detection toggle button
        const detectionToggle = document.getElementById('detectionToggle');
        if (detectionToggle) {
            detectionToggle.addEventListener('click', () => {
                this.toggleDetection();
            });
        }
        
        // Employee form
        const employeeForm = document.getElementById('employeeForm');
        if (employeeForm) {
            employeeForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.addEmployee();
            });
        }
        
        // Face upload
        const faceUpload = document.getElementById('faceUpload');
        if (faceUpload) {
            faceUpload.addEventListener('change', (e) => {
                this.handleFaceUpload(e);
            });
        }
        
        // Refresh buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-action="refresh-employees"]')) {
                this.loadEmployees();
            }
            
            if (e.target.matches('[data-action="delete-employee"]')) {
                const employeeId = e.target.dataset.employeeId;
                this.deleteEmployee(employeeId);
            }
            
            if (e.target.matches('[data-action="request-frame"]')) {
                this.requestCurrentFrame();
            }
        });
    }
    
    registerAsDesktop() {
        this.socket.emit('register_desktop_client', {
            user_agent: navigator.userAgent,
            screen_size: `${screen.width}x${screen.height}`,
            type: 'desktop_monitor'
        });
    }
    
    toggleDetection() {
        const newState = !this.detectionActive;
        this.socket.emit('toggle_detection', { active: newState });
        
        // Update UI immediately for responsiveness
        const toggle = document.getElementById('detectionToggle');
        if (toggle) {
            toggle.disabled = true;
            setTimeout(() => {
                toggle.disabled = false;
            }, 1000);
        }
    }
    
    updateDetectionToggle() {
        const toggle = document.getElementById('detectionToggle');
        if (toggle) {
            if (this.detectionActive) {
                toggle.textContent = '🛑 Stop Detection';
                toggle.className = 'detection-toggle active';
            } else {
                toggle.textContent = '▶️ Start Detection';
                toggle.className = 'detection-toggle inactive';
            }
        }
    }
    
    handleVideoFrame(data) {
        const videoDisplay = document.getElementById('videoDisplay');
        const frameImg = document.getElementById('currentFrame');
        
        if (frameImg && data.frame) {
            frameImg.src = data.frame;
            frameImg.style.display = 'block';
            
            // Update frame timestamp
            const timestamp = document.getElementById('frameTimestamp');
            if (timestamp) {
                timestamp.textContent = new Date(data.timestamp * 1000).toLocaleTimeString();
            }
        }
        
        // Update detection overlay
        this.updateDetectionOverlay(data.faces || []);
        
        // Update stats
        this.stats.totalFrames++;
        if (data.faces && data.faces.length > 0) {
            this.stats.totalDetections += data.faces.length;
        }
        
        this.updateStatsDisplay();
        
        // Hide "no video" message
        const noVideoMsg = document.getElementById('noVideoMessage');
        if (noVideoMsg) {
            noVideoMsg.style.display = 'none';
        }
    }
    
    updateDetectionOverlay(faces) {
        // Clear existing overlays
        const existingOverlays = document.querySelectorAll('.face-box');
        existingOverlays.forEach(overlay => overlay.remove());
        
        if (faces.length === 0) return;
        
        const videoContainer = document.getElementById('videoDisplay');
        const frameImg = document.getElementById('currentFrame');
        
        if (!videoContainer || !frameImg) return;
        
        faces.forEach((face, index) => {
            const faceBox = document.createElement('div');
            faceBox.className = 'face-box';
            faceBox.style.left = `${face.face.x}px`;
            faceBox.style.top = `${face.face.y}px`;
            faceBox.style.width = `${face.face.w}px`;
            faceBox.style.height = `${face.face.h}px`;
            
            if (face.employee) {
                const label = document.createElement('div');
                label.className = 'face-label';
                label.textContent = `${face.employee.name} (${Math.round(face.confidence * 100)}%)`;
                faceBox.appendChild(label);
            }
            
            videoContainer.appendChild(faceBox);
        });
    }
    
    updateConnectionStatus(connected) {
        const statusDot = document.getElementById('connectionStatus');
        const statusText = document.getElementById('connectionText');
        
        if (statusDot) {
            statusDot.className = `status-dot ${connected ? 'connected' : 'disconnected'}`;
        }
        
        if (statusText) {
            statusText.textContent = connected ? 'Connected' : 'Disconnected';
        }
    }
    
    requestCurrentFrame() {
        this.socket.emit('get_current_frame');
    }
    
    async loadEmployees() {
        try {
            const response = await fetch('/api/employees');
            const data = await response.json();
            
            if (data.success) {
                this.employees = data.data;
                this.renderEmployees();
            } else {
                this.showNotification('Failed to load employees', 'error');
            }
        } catch (error) {
            console.error('Error loading employees:', error);
            this.showNotification('Error loading employees', 'error');
        }
    }
    
    renderEmployees() {
        const container = document.getElementById('employeesList');
        if (!container) return;
        
        container.innerHTML = '';
        
        this.employees.forEach(employee => {
            const card = document.createElement('div');
            card.className = 'employee-card';
            card.innerHTML = `
                <div class="employee-info">
                    <div class="employee-name">${employee.name}</div>
                    <div class="employee-details">
                        <div>Code: ${employee.employee_code}</div>
                        <div>Department: ${employee.department}</div>
                        <div>Position: ${employee.position}</div>
                        <div>Images: ${employee.image_count || 0}</div>
                    </div>
                </div>
                <div class="employee-actions">
                    <button class="btn btn-secondary" data-action="delete-employee" data-employee-id="${employee.id}">
                        Delete
                    </button>
                </div>
            `;
            
            container.appendChild(card);
        });
    }
    
    async addEmployee() {
        const form = document.getElementById('employeeForm');
        const formData = new FormData(form);
        
        const employeeData = {
            name: formData.get('name'),
            employee_code: formData.get('employee_code'),
            department: formData.get('department'),
            position: formData.get('position'),
            email: formData.get('email'),
            phone: formData.get('phone')
        };
        
        try {
            const response = await fetch('/api/employees', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(employeeData)
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Employee added successfully', 'success');
                form.reset();
                this.loadEmployees();
            } else {
                this.showNotification(data.error || 'Failed to add employee', 'error');
            }
        } catch (error) {
            console.error('Error adding employee:', error);
            this.showNotification('Error adding employee', 'error');
        }
    }
    
    async deleteEmployee(employeeId) {
        if (!confirm('Are you sure you want to delete this employee?')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/employees/${employeeId}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Employee deleted successfully', 'success');
                this.loadEmployees();
            } else {
                this.showNotification(data.error || 'Failed to delete employee', 'error');
            }
        } catch (error) {
            console.error('Error deleting employee:', error);
            this.showNotification('Error deleting employee', 'error');
        }
    }
    
    async handleFaceUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const employeeId = document.getElementById('selectedEmployee').value;
        if (!employeeId) {
            this.showNotification('Please select an employee first', 'warning');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('employee_id', employeeId);
        
        try {
            const response = await fetch('/api/upload-face', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Face image uploaded successfully', 'success');
                this.loadEmployees(); // Refresh to update image count
            } else {
                this.showNotification(data.error || 'Failed to upload face image', 'error');
            }
        } catch (error) {
            console.error('Error uploading face:', error);
            this.showNotification('Error uploading face image', 'error');
        }
    }
    
    updateStatsDisplay() {
        const updates = {
            'totalFrames': this.stats.totalFrames,
            'totalDetections': this.stats.totalDetections,
            'mobileClients': this.stats.mobileClients,
            'desktopClients': this.stats.desktopClients
        };
        
        Object.entries(updates).forEach(([key, value]) => {
            const element = document.getElementById(key);
            if (element) {
                element.textContent = value;
            }
        });
    }
    
    async loadSystemStats() {
        try {
            const response = await fetch('/api/stats');
            const data = await response.json();
            
            if (data.success) {
                this.stats = { ...this.stats, ...data.data };
                this.updateStatsDisplay();
            }
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    }
    
    startStatsUpdate() {
        // Update stats every 5 seconds
        setInterval(() => {
            this.loadSystemStats();
        }, 5000);
        
        // Initial load
        this.loadSystemStats();
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type}`;
        notification.textContent = message;
        notification.style.position = 'fixed';
        notification.style.top = '20px';
        notification.style.right = '20px';
        notification.style.zIndex = '9999';
        notification.style.minWidth = '250px';
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
}

// Initialize desktop monitor when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.desktopMonitor = new DesktopMonitor();
});

// Export for potential external use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DesktopMonitor;
}