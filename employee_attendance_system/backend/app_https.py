#!/usr/bin/env python3
"""
HTTPS Server for Mobile Camera Access
Khắc phục lỗi Mobile không thể mở camera do yêu cầu HTTPS
"""
import ssl
import os
import sys
from app import app, socketio

def main():
    print("🔐 Starting HTTPS server for mobile camera access...")
    print("=" * 60)
    
    try:
        # Check if pyOpenSSL is available
        try:
            import OpenSSL
            print("✅ OpenSSL library found")
        except ImportError:
            print("❌ OpenSSL library not found")
            print("💡 Install with: pip install pyOpenSSL")
            sys.exit(1)
        
        print("🚀 Starting HTTPS server...")
        print("📱 Mobile Interface: https://localhost:5000/mobile")
        print("🖥️  Desktop Monitor: https://localhost:5000/")
        print("👥 Management: https://localhost:5000/management")
        print("=" * 60)
        print("⚠️  IMPORTANT: Accept the security warning in your browser")
        print("    (Self-signed certificate for development)")
        print("=" * 60)
        
        # Start HTTPS server with adhoc SSL
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for better performance
            ssl_context='adhoc',  # Self-signed certificate
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 HTTPS server stopped by user")
    except Exception as e:
        print(f"❌ HTTPS server error: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Install pyOpenSSL: pip install pyOpenSSL")
        print("2. Try regular HTTP server: python app.py")
        print("3. Check if port 5000 is available")
    finally:
        print("👋 BHK Tech Attendance System HTTPS shutdown complete")

if __name__ == '__main__':
    main()