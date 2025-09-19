/**
 * employee-management.js - EMPLOYEE MANAGEMENT UTILITIES
 * Handles employee CRUD operations, face image management, and data validation
 */

class EmployeeManager {
    constructor() {
        this.employees = [];
        this.currentEmployee = null;
        this.uploadedImages = [];
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadEmployees();
        
        console.log('👥 Employee Manager initialized');
    }
    
    setupEventListeners() {
        // Employee form submission
        const employeeForm = document.getElementById('employeeForm');
        if (employeeForm) {
            employeeForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleEmployeeSubmit(e);
            });
        }
        
        // Face image upload
        const faceUploadInput = document.getElementById('faceUpload');
        if (faceUploadInput) {
            faceUploadInput.addEventListener('change', (e) => {
                this.handleFaceUpload(e);
            });
        }
        
        // Drag and drop for face upload
        const uploadArea = document.getElementById('faceUploadArea');
        if (uploadArea) {
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                this.handleFileDrop(e);
            });
            
            uploadArea.addEventListener('click', () => {
                faceUploadInput.click();
            });
        }
        
        // Employee selection for face upload
        const employeeSelect = document.getElementById('selectedEmployee');
        if (employeeSelect) {
            employeeSelect.addEventListener('change', (e) => {
                this.currentEmployee = e.target.value;
                this.updateUploadArea();
            });
        }
        
        // Form validation
        const inputs = document.querySelectorAll('#employeeForm input');
        inputs.forEach(input => {
            input.addEventListener('blur', () => {
                this.validateInput(input);
            });
            
            input.addEventListener('input', () => {
                this.clearInputError(input);
            });
        });
        
        // Employee actions (edit, delete, view)
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-action="edit-employee"]')) {
                const employeeId = e.target.dataset.employeeId;
                this.editEmployee(employeeId);
            }
            
            if (e.target.matches('[data-action="delete-employee"]')) {
                const employeeId = e.target.dataset.employeeId;
                this.confirmDeleteEmployee(employeeId);
            }
            
            if (e.target.matches('[data-action="view-employee"]')) {
                const employeeId = e.target.dataset.employeeId;
                this.viewEmployee(employeeId);
            }
            
            if (e.target.matches('[data-action="add-employee"]')) {
                this.showEmployeeForm();
            }
            
            if (e.target.matches('[data-action="cancel-form"]')) {
                this.hideEmployeeForm();
            }
        });
    }
    
    async loadEmployees() {
        try {
            this.showLoading('Loading employees...');
            
            const response = await fetch('/api/employees');
            const data = await response.json();
            
            if (data.success) {
                this.employees = data.data;
                this.renderEmployeesList();
                this.updateEmployeeSelect();
                this.showNotification('Employees loaded successfully', 'success');
            } else {
                this.showNotification(data.error || 'Failed to load employees', 'error');
            }
        } catch (error) {
            console.error('Error loading employees:', error);
            this.showNotification('Error loading employees', 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    renderEmployeesList() {
        const container = document.getElementById('employeesList');
        if (!container) return;
        
        if (this.employees.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <h3>No Employees Found</h3>
                    <p>Get started by adding your first employee.</p>
                    <button class="btn btn-primary" data-action="add-employee">
                        Add First Employee
                    </button>
                </div>
            `;
            return;
        }
        
        container.innerHTML = '';
        
        this.employees.forEach(employee => {
            const card = this.createEmployeeCard(employee);
            container.appendChild(card);
        });
    }
    
    createEmployeeCard(employee) {
        const card = document.createElement('div');
        card.className = 'employee-card';
        card.dataset.employeeId = employee.id;
        
        card.innerHTML = `
            <div class="employee-avatar">
                ${this.getEmployeeInitials(employee.name)}
            </div>
            <div class="employee-info">
                <div class="employee-name">${this.escapeHtml(employee.name)}</div>
                <div class="employee-details">
                    <div><strong>Code:</strong> ${employee.employee_code}</div>
                    <div><strong>Department:</strong> ${employee.department}</div>
                    <div><strong>Position:</strong> ${employee.position}</div>
                    <div><strong>Email:</strong> ${employee.email || 'N/A'}</div>
                    <div><strong>Phone:</strong> ${employee.phone || 'N/A'}</div>
                </div>
                <div class="employee-stats">
                    <span class="stat-badge">
                        <span class="stat-value">${employee.image_count || 0}</span>
                        <span class="stat-label">Face Images</span>
                    </span>
                    <span class="status status-${employee.status || 'active'}">
                        ${employee.status || 'active'}
                    </span>
                </div>
            </div>
            <div class="employee-actions">
                <button class="btn btn-sm btn-secondary" data-action="view-employee" data-employee-id="${employee.id}">
                    View
                </button>
                <button class="btn btn-sm btn-primary" data-action="edit-employee" data-employee-id="${employee.id}">
                    Edit
                </button>
                <button class="btn btn-sm btn-danger" data-action="delete-employee" data-employee-id="${employee.id}">
                    Delete
                </button>
            </div>
        `;
        
        return card;
    }
    
    updateEmployeeSelect() {
        const select = document.getElementById('selectedEmployee');
        if (!select) return;
        
        select.innerHTML = '<option value="">Select Employee for Face Upload</option>';
        
        this.employees.forEach(employee => {
            const option = document.createElement('option');
            option.value = employee.id;
            option.textContent = `${employee.name} (${employee.employee_code})`;
            select.appendChild(option);
        });
    }
    
    async handleEmployeeSubmit(event) {
        event.preventDefault();
        
        const form = event.target;
        const formData = new FormData(form);
        
        // Validate form
        if (!this.validateForm(form)) {
            return;
        }
        
        const employeeData = {
            name: formData.get('name').trim(),
            employee_code: formData.get('employee_code').trim(),
            department: formData.get('department').trim(),
            position: formData.get('position').trim(),
            email: formData.get('email').trim(),
            phone: formData.get('phone').trim()
        };
        
        try {
            this.showLoading('Saving employee...');
            
            const isEdit = form.dataset.employeeId;
            const url = isEdit ? `/api/employees/${form.dataset.employeeId}` : '/api/employees';
            const method = isEdit ? 'PUT' : 'POST';
            
            const response = await fetch(url, {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(employeeData)
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(
                    isEdit ? 'Employee updated successfully' : 'Employee added successfully',
                    'success'
                );
                this.hideEmployeeForm();
                this.loadEmployees();
            } else {
                this.showNotification(data.error || 'Failed to save employee', 'error');
            }
        } catch (error) {
            console.error('Error saving employee:', error);
            this.showNotification('Error saving employee', 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    async handleFaceUpload(event) {
        const files = Array.from(event.target.files);
        if (files.length === 0) return;
        
        if (!this.currentEmployee) {
            this.showNotification('Please select an employee first', 'warning');
            return;
        }
        
        for (const file of files) {
            await this.uploadFaceImage(file, this.currentEmployee);
        }
        
        // Refresh employee list to update image count
        this.loadEmployees();
    }
    
    async handleFileDrop(event) {
        const files = Array.from(event.dataTransfer.files);
        const imageFiles = files.filter(file => file.type.startsWith('image/'));
        
        if (imageFiles.length === 0) {
            this.showNotification('Please drop image files only', 'warning');
            return;
        }
        
        if (!this.currentEmployee) {
            this.showNotification('Please select an employee first', 'warning');
            return;
        }
        
        for (const file of imageFiles) {
            await this.uploadFaceImage(file, this.currentEmployee);
        }
        
        this.loadEmployees();
    }
    
    async uploadFaceImage(file, employeeId) {
        if (!this.validateImageFile(file)) {
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('employee_id', employeeId);
        
        try {
            this.showLoading(`Uploading ${file.name}...`);
            
            const response = await fetch('/api/upload-face', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(`${file.name} uploaded successfully`, 'success');
                this.uploadedImages.push({
                    id: data.data.id,
                    filename: file.name,
                    employeeId: employeeId
                });
            } else {
                this.showNotification(`Failed to upload ${file.name}: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('Error uploading face image:', error);
            this.showNotification(`Error uploading ${file.name}`, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    validateImageFile(file) {
        // Check file type
        if (!file.type.startsWith('image/')) {
            this.showNotification('Please select an image file', 'error');
            return false;
        }
        
        // Check file size (max 5MB)
        const maxSize = 5 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showNotification('Image file too large (max 5MB)', 'error');
            return false;
        }
        
        return true;
    }
    
    validateForm(form) {
        const inputs = form.querySelectorAll('input[required]');
        let isValid = true;
        
        inputs.forEach(input => {
            if (!this.validateInput(input)) {
                isValid = false;
            }
        });
        
        return isValid;
    }
    
    validateInput(input) {
        const value = input.value.trim();
        const type = input.type;
        const name = input.name;
        
        this.clearInputError(input);
        
        // Required field validation
        if (input.hasAttribute('required') && !value) {
            this.showInputError(input, 'This field is required');
            return false;
        }
        
        // Email validation
        if (type === 'email' && value) {
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(value)) {
                this.showInputError(input, 'Please enter a valid email address');
                return false;
            }
        }
        
        // Phone validation
        if (name === 'phone' && value) {
            const phoneRegex = /^[\d\s\-\+\(\)]+$/;
            if (!phoneRegex.test(value)) {
                this.showInputError(input, 'Please enter a valid phone number');
                return false;
            }
        }
        
        // Employee code validation
        if (name === 'employee_code' && value) {
            if (value.length < 3) {
                this.showInputError(input, 'Employee code must be at least 3 characters');
                return false;
            }
        }
        
        return true;
    }
    
    showInputError(input, message) {
        input.classList.add('error');
        
        let errorDiv = input.parentNode.querySelector('.input-error');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.className = 'input-error';
            input.parentNode.appendChild(errorDiv);
        }
        
        errorDiv.textContent = message;
    }
    
    clearInputError(input) {
        input.classList.remove('error');
        const errorDiv = input.parentNode.querySelector('.input-error');
        if (errorDiv) {
            errorDiv.remove();
        }
    }
    
    editEmployee(employeeId) {
        const employee = this.employees.find(emp => emp.id == employeeId);
        if (!employee) return;
        
        this.showEmployeeForm(employee);
    }
    
    viewEmployee(employeeId) {
        const employee = this.employees.find(emp => emp.id == employeeId);
        if (!employee) return;
        
        // Create and show employee detail modal
        this.showEmployeeDetail(employee);
    }
    
    confirmDeleteEmployee(employeeId) {
        const employee = this.employees.find(emp => emp.id == employeeId);
        if (!employee) return;
        
        if (confirm(`Are you sure you want to delete ${employee.name}?\nThis action cannot be undone.`)) {
            this.deleteEmployee(employeeId);
        }
    }
    
    async deleteEmployee(employeeId) {
        try {
            this.showLoading('Deleting employee...');
            
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
        } finally {
            this.hideLoading();
        }
    }
    
    showEmployeeForm(employee = null) {
        const form = document.getElementById('employeeForm');
        if (!form) return;
        
        // Reset form
        form.reset();
        this.clearAllInputErrors();
        
        if (employee) {
            // Populate form for editing
            form.dataset.employeeId = employee.id;
            form.querySelector('[name="name"]').value = employee.name;
            form.querySelector('[name="employee_code"]').value = employee.employee_code;
            form.querySelector('[name="department"]').value = employee.department;
            form.querySelector('[name="position"]').value = employee.position;
            form.querySelector('[name="email"]').value = employee.email || '';
            form.querySelector('[name="phone"]').value = employee.phone || '';
        } else {
            // Clear edit mode
            delete form.dataset.employeeId;
        }
        
        // Show form section
        const formSection = document.getElementById('employeeFormSection');
        if (formSection) {
            formSection.style.display = 'block';
            form.scrollIntoView({ behavior: 'smooth' });
        }
    }
    
    hideEmployeeForm() {
        const formSection = document.getElementById('employeeFormSection');
        if (formSection) {
            formSection.style.display = 'none';
        }
        
        const form = document.getElementById('employeeForm');
        if (form) {
            form.reset();
            delete form.dataset.employeeId;
        }
        
        this.clearAllInputErrors();
    }
    
    clearAllInputErrors() {
        const errorInputs = document.querySelectorAll('input.error');
        const errorMessages = document.querySelectorAll('.input-error');
        
        errorInputs.forEach(input => input.classList.remove('error'));
        errorMessages.forEach(msg => msg.remove());
    }
    
    updateUploadArea() {
        const uploadArea = document.getElementById('faceUploadArea');
        if (!uploadArea) return;
        
        if (this.currentEmployee) {
            const employee = this.employees.find(emp => emp.id == this.currentEmployee);
            if (employee) {
                uploadArea.innerHTML = `
                    <div class="upload-icon">📸</div>
                    <div class="upload-text">
                        <p>Upload face images for <strong>${employee.name}</strong></p>
                        <p>Drop images here or click to select</p>
                        <small>Supported: JPG, PNG, WEBP (Max 5MB each)</small>
                    </div>
                `;
            }
        } else {
            uploadArea.innerHTML = `
                <div class="upload-icon">👤</div>
                <div class="upload-text">
                    <p>Select an employee first</p>
                    <small>Then you can upload face images</small>
                </div>
            `;
        }
    }
    
    getEmployeeInitials(name) {
        return name
            .split(' ')
            .map(word => word.charAt(0))
            .join('')
            .substring(0, 2)
            .toUpperCase();
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    showLoading(message = 'Loading...') {
        // Implementation depends on your loading component
        console.log('Loading:', message);
    }
    
    hideLoading() {
        // Implementation depends on your loading component
        console.log('Loading complete');
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            min-width: 250px;
            max-width: 400px;
            animation: slideIn 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
}

// Initialize employee manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.employeeManager = new EmployeeManager();
});

// Export for potential external use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EmployeeManager;
}