#!/bin/bash

echo "🚀 Bắt đầu cài đặt hệ thống điểm danh..."

# Kiểm tra Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 chưa được cài đặt"
    exit 1
fi

echo "✅ Python3 đã có sẵn"

# Tạo virtual environment
echo "📦 Tạo virtual environment..."
python3 -m venv venv

# Kích hoạt virtual environment
source venv/bin/activate

# Upgrade pip
echo "⬆️ Cập nhật pip..."
pip install --upgrade pip

# Cài đặt requirements
echo "📥 Cài đặt dependencies..."
pip install -r requirements.txt

# Chạy setup
echo "⚙️ Thiết lập project..."
python setup.py

echo "🎉 Cài đặt hoàn tất!"
echo ""
echo "🚀 Để chạy ứng dụng:"
echo "1. Kích hoạt virtual environment: source venv/bin/activate"
echo "2. Chạy server: python app.py"
echo "3. Mở trình duyệt: http://localhost:5000"
echo "4. Mobile app: http://localhost:5000/mobile"