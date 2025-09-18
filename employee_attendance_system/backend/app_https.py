#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🔐 BHK Tech - HTTPS Server for Mobile Camera Access
File: backend/app_https.py
"""

import os
import sys
import ssl
from flask import Flask
from flask_socketio import SocketIO

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the existing app
from app import app, socketio

def create_ssl_context():
    """Create SSL context for HTTPS"""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    
    # Try to load certificate files
    cert_file = os.path.join(os.path.dirname(__file__), 'cert.cer')
    key_file = os.path.join(os.path.dirname(__file__), 'cert.pfx')
    
    if os.path.exists(cert_file) and os.path.exists(key_file):
        try:
            # For self-signed certificate
            context.load_cert_chain(cert_file, key_file, password='bhktech2024')
            print("✅ SSL certificate loaded successfully")
            return context
        except Exception as e:
            print(f"❌ Failed to load SSL certificate: {e}")
            print("💡 Using adhoc SSL context")
            return 'adhoc'
    else:
        print("⚠️ Certificate files not found, using adhoc SSL")
        return 'adhoc'

if __name__ == '__main__':
    print("🔐 Starting BHK Tech Attendance System - HTTPS Mode")
    print("=" * 50)
    
    # Configure for HTTPS
    app.config['HTTPS'] = True
    
    # Create SSL context
    ssl_context = create_ssl_context()
    
    # Start server with HTTPS
    try:
        print("🚀 Server starting on https://localhost:5000")
        print("📱 Mobile access: https://[your-ip]:5000/mobile")
        print("🖥️ Desktop access: https://localhost:5000")
        print("⚠️ Warning: Self-signed certificate will show security warning")
        print("💡 Accept the security warning to continue")
        print("=" * 50)
        
        socketio.run(
            app,
            host='0.0.0.0',  # Allow external connections
            port=5000,
            debug=False,
            ssl_context=ssl_context,
            certfile=None,
            keyfile=None,
            allow_unsafe_werkzeug=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start HTTPS server: {e}")
        print("💡 Trying fallback HTTPS mode...")
        
        try:
            # Fallback with simpler SSL
            socketio.run(
                app,
                host='0.0.0.0',
                port=5000,
                debug=False,
                ssl_context='adhoc',
                allow_unsafe_werkzeug=True
            )
        except Exception as e2:
            print(f"❌ Fallback HTTPS also failed: {e2}")
            print("💡 Please install pyOpenSSL: pip install pyOpenSSL")
            print("💡 Or use HTTP mode for testing: python app.py")
            sys.exit(1)