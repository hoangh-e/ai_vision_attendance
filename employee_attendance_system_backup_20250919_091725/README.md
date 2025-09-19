# Hệ thống điểm danh nhân viên với Face Recognition

Hệ thống điểm danh tự động sử dụng công nghệ nhận diện khuôn mặt, WebRTC cho truyền video realtime từ thiết bị di động đến laptop xử lý.

## Tính năng chính

### 🔧 Laptop Interface (Server)
- **Tab 1 - Quản lý nhân viên**: CRUD nhân viên, upload/quản lý ảnh (tối đa 10 ảnh/người)
- **Tab 2 - Phát hiện realtime**: Nhận video từ mobile, xử lý AI, hiển thị kết quả
- **Tab 3 - Chụp ảnh từ laptop**: Chụp ảnh trực tiếp từ webcam laptop để bổ sung dữ liệu

### 📱 Mobile App
- Gửi video realtime qua WebRTC đến laptop
- Nhận kết quả nhận diện và hiển thị bounding box
- Giao diện tối ưu cho thiết bị di động
- Hỗ trợ chuyển đổi camera trước/sau

### 🤖 AI Processing
- Sử dụng DeepFace với model Facenet512
- Vector hóa khuôn mặt và lưu trữ trong SQLite
- So sánh realtime với độ chính xác cao
- Hỗ trợ nhiều khuôn mặt trong một frame

### 🗄️ Database
- SQLite với 3 bảng: employees, vector_face, attendance_logs
- Lưu trữ thông tin nhân viên và vector đặc trưng khuôn mặt
- Tự động backup và quản lý dữ liệu

## Cấu trúc Project

```
employee_attendance_system/
├── backend/
│   ├── app.py                 # Flask server chính
│   ├── config.py              # Cấu hình
│   ├── database/              # Database models và connection
│   ├── services/              # Logic xử lý (Face, Employee, WebRTC)
│   ├── api/                   # API routes
│   ├── static/                # Static files và uploads
│   └── templates/             # HTML templates
├── mobile_app/                # Mobile interface
├── requirements.txt           # Python dependencies
├── setup.py                  # Script thiết lập
└── README.md
```

## Công nghệ sử dụng

- **Backend**: Flask, Flask-SocketIO, SQLAlchemy
- **AI/ML**: DeepFace, TensorFlow, OpenCV
- **Frontend**: HTML5, JavaScript, Bootstrap
- **Database**: SQLite
- **Communication**: WebRTC, Socket.IO
- **Mobile**: Progressive Web App (PWA)

## Cài đặt và chạy

### Yêu cầu hệ thống
- **Laptop/Server**: Python 3.8+, 4GB RAM trở lên (cho xử lý AI)
- **Mobile**: Android 6.0+ hoặc iOS 12+, camera trước/sau
- **Mạng**: WiFi LAN chung giữa laptop và mobile

### 🚀 HƯỚNG DẪN CÀI ĐẶT THEO PLATFORM

## 📱 MOBILE APP (Android/iOS)

### Bước 1: Chuẩn bị môi trường
```bash
# Đảm bảo laptop và mobile cùng mạng WiFi
# Kiểm tra IP laptop: ipconfig (Windows) hoặc ifconfig (Linux/Mac)
```

### Bước 2: Thiết lập kết nối
1. **Mở Chrome/Safari trên điện thoại**
2. **Truy cập:** `http://[IP_LAPTOP]:5000/mobile`
   - Ví dụ: `http://192.168.1.100:5000/mobile`
3. **Cho phép quyền camera** khi trình duyệt hỏi

### Bước 3: Cấu hình Mobile App
1. **Bấm nút "Cài đặt"** trong app
2. **Điều chỉnh:**
   - Server URL: `[IP_LAPTOP]:5000`  
   - Độ phân giải: `640x480` (cho mạng chậm) hoặc `1280x720` (HD)
   - FPS gửi: `10 FPS` (khuyến nghị)
3. **Lưu cài đặt**

### Bước 4: Sử dụng Mobile App
1. **Bấm "Bật camera"** để khởi động
2. **Đổi camera** trước/sau nếu cần
3. **Kiểm tra trạng thái kết nối** (chấm xanh = OK)
4. **Xem kết quả nhận diện** ở phía dưới màn hình

### Cài đặt tự động (chỉ dành cho laptop/server)

**Linux/Mac:**
```bash
chmod +x install.sh
./install.sh
```

**Windows:**
```cmd
install.bat
```

## 💻 LAPTOP/SERVER SETUP

### Phương pháp 1: Cài đặt tự động (Khuyến nghị)
```bash
# Windows
.\install.bat

# Linux/Mac  
chmod +x install.sh && ./install.sh
```

### Phương pháp 2: Cài đặt thủ công

#### Bước 1: Chuẩn bị môi trường
```bash
# Kiểm tra Python (cần 3.8+)
python --version

# Clone project
git clone <project_url>
cd employee_attendance_system
```

#### Bước 2: Tạo Virtual Environment
```bash
# Tạo venv
python -m venv venv

# Kích hoạt
# Windows:
venv\Scripts\activate.bat

# Linux/Mac:
source venv/bin/activate
```

#### Bước 3: Cài đặt Dependencies
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Cài đặt packages
pip install -r requirements.txt
```

#### Bước 4: Khởi tạo Database
```bash
python setup.py
```

#### Bước 5: Chạy Server
```bash
# Chạy ở mode development
python backend/app.py

# Hoặc chạy ở root folder
cd backend && python app.py
```

#### Bước 6: Kiểm tra kết nối
- **Laptop interface:** http://localhost:5000
- **Mobile access:** http://[YOUR_IP]:5000/mobile
- **Lấy IP:** `ipconfig` (Windows) hoặc `ifconfig` (Linux/Mac)

## 🎯 HƯỚNG DẪN SỬ DỤNG THEO THỨ TỰ

### Phase 1: Khởi động hệ thống
1. **Khởi động server trên laptop**
   ```bash
   cd backend
   python app.py
   ```
2. **Kiểm tra:** http://localhost:5000 (trên laptop)
3. **Lấy IP laptop:** `ipconfig` hoặc `ifconfig`

### Phase 2: Thiết lập nhân viên  
1. **Truy cập:** http://localhost:5000 trên laptop
2. **Tab "Quản lý nhân viên":**
   - Thêm nhân viên mới (tên, mã NV, phòng ban)
   - **Upload 5-10 ảnh** cho mỗi nhân viên (nhiều góc độ, ánh sáng khác nhau)
   - Kiểm tra số ảnh đã upload

### Phase 3: Kết nối Mobile
1. **Mở điện thoại, truy cập:** `http://[IP_LAPTOP]:5000/mobile`
2. **Cài đặt Mobile:**
   - Server URL: `[IP_LAPTOP]:5000`
   - Độ phân giải: 1280x720
   - FPS: 10
3. **Bật camera và kiểm tra kết nối** (chấm xanh)

### Phase 4: Bắt đầu điểm danh
1. **Trên laptop:** Tab "Phát hiện realtime"
2. **Bật "Bắt đầu nhận camera"**
3. **Bật "Bắt đầu detect"**  
4. **Trên mobile:** Bật camera và đưa khuôn mặt vào khung hình
5. **Xem kết quả** realtime trên cả laptop và mobile

## Cấu hình nâng cao

### Chỉnh sửa model AI
```python
# Trong config.py
FACE_RECOGNITION_MODEL = 'Facenet512'  # hoặc 'VGG-Face', 'Facenet', 'OpenFace'
FACE_DETECTION_BACKEND = 'opencv'      # hoặc 'mtcnn', 'ssd', 'dlib'
FACE_RECOGNITION_THRESHOLD = 0.6       # Ngưỡng nhận diện (thấp hơn = nghiêm ngặt hơn)
```

### Cấu hình mạng
```python
# Trong app.py
socketio.run(app, host='0.0.0.0', port=5000)  # Cho phép truy cập từ mạng LAN
```

## Troubleshooting

### Lỗi thường gặp

1. **Không kết nối được camera:**
   - Kiểm tra quyền truy cập camera trên browser
   - Thử chuyển đổi camera trước/sau
   - Kiểm tra camera có đang được sử dụng bởi app khác

2. **Lỗi cài đặt TensorFlow:**
   ```bash
   # Thử cài đặt version cụ thể
   pip install tensorflow==2.13.0
   ```

3. **Mobile không kết nối được:**
   - Kiểm tra laptop và mobile cùng mạng WiFi
   - Thử truy cập bằng IP của laptop: http://192.168.x.x:5000/mobile
   - Tắt firewall tạm thời để test

4. **AI xử lý chậm:**
   - Giảm resolution camera trong settings mobile
   - Giảm FPS từ 30 xuống 10-15
   - Đóng các ứng dụng khác để giải phóng RAM

### Performance tuning

1. **Tối ưu cho GPU (nếu có):**
```python
# Thêm vào đầu app.py
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

2. **Tối ưu database:**
```sql
-- Thêm index để tăng tốc truy vấn
CREATE INDEX idx_employee_code ON employees(employee_code);
CREATE INDEX idx_vector_employee_id ON vector_face(employee_id);
CREATE INDEX idx_attendance_employee_id ON attendance_logs(employee_id);
```

## API Documentation

### Employee Management
```http
GET /api/employees
POST /api/employees
PUT /api/employees/{id}
DELETE /api/employees/{id}
```

### Face Vector Management
```http
POST /api/employees/{id}/upload
GET /api/employees/{id}/vectors
DELETE /api/vectors/{id}
```

### Real-time Events (Socket.IO)
```javascript
// Client events
socket.emit('camera_frame', {image: base64_data});
socket.emit('toggle_detection', {active: true});

// Server events
socket.on('detection_result', function(data) {
    // Xử lý kết quả nhận diện
});
```

## ⚡ QUICK START - CHỈ 3 BƯỚC

### Laptop (Server):
```bash
# Bước 1: Cài đặt
.\install.bat

# Bước 2: Chạy server  
cd backend && python app.py

# Bước 3: Mở http://localhost:5000
```

### Mobile (Client):
```bash
# Bước 1: Lấy IP laptop
ipconfig  # Windows
ifconfig  # Linux/Mac

# Bước 2: Mở trình duyệt mobile
# Truy cập: http://[IP_LAPTOP]:5000/mobile

# Bước 3: Cho phép camera và bắt đầu
```

### Tóm tắt URLs:
- **Laptop interface:** http://localhost:5000
- **Mobile app:** http://[YOUR_IP]:5000/mobile  
- **Ví dụ mobile:** http://192.168.1.100:5000/mobile

**🎉 Chúc bạn triển khai thành công!**

---
*Project được phát triển với ❤️ bởi **BHK Tech***
