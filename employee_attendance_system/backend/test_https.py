#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🧪 Test HTTPS Configuration
File: backend/test_https.py
"""

import requests
import ssl
import socket
from urllib3.exceptions import InsecureRequestWarning
import warnings

# Suppress SSL warnings for self-signed certificates
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

def test_https_connection():
    """Test if HTTPS server is accessible"""
    print("🧪 Testing HTTPS Configuration")
    print("=" * 40)
    
    base_url = "https://localhost:5000"
    
    try:
        # Test basic HTTPS connection
        print("1. Testing basic HTTPS connection...")
        response = requests.get(base_url, verify=False, timeout=10)
        
        if response.status_code == 200:
            print("✅ HTTPS server is accessible")
            print(f"   Status: {response.status_code}")
            print(f"   URL: {base_url}")
        else:
            print(f"⚠️ Server responded with status: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - Server not running")
        return False
    except Exception as e:
        print(f"❌ HTTPS test failed: {e}")
        return False
    
    try:
        # Test mobile endpoint
        print("2. Testing mobile endpoint...")
        mobile_response = requests.get(f"{base_url}/mobile", verify=False, timeout=10)
        
        if mobile_response.status_code == 200:
            print("✅ Mobile endpoint is accessible")
        else:
            print(f"⚠️ Mobile endpoint status: {mobile_response.status_code}")
            
    except Exception as e:
        print(f"❌ Mobile endpoint test failed: {e}")
    
    try:
        # Test SSL certificate info
        print("3. Testing SSL certificate...")
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        
        with socket.create_connection(('localhost', 5000), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname='localhost') as ssock:
                cert = ssock.getpeercert()
                print("✅ SSL certificate is working")
                if cert:
                    print(f"   Subject: {cert.get('subject', 'N/A')}")
                    print(f"   Issuer: {cert.get('issuer', 'N/A')}")
                else:
                    print("   Using self-signed certificate")
                    
    except Exception as e:
        print(f"❌ SSL certificate test failed: {e}")
    
    print("\n📱 Mobile Access Instructions:")
    print("=" * 40)
    print("1. Connect your mobile device to the same network")
    print("2. Find your computer's IP address:")
    print("   - Windows: ipconfig")
    print("   - Mac/Linux: ifconfig")
    print("3. Access: https://[YOUR-IP]:5000/mobile")
    print("4. Accept the security warning (self-signed certificate)")
    print("5. Grant camera permissions when prompted")
    
    return True

if __name__ == "__main__":
    test_https_connection()