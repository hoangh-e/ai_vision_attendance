#!/usr/bin/env python3
"""
main_app.py - BHK TECH ATTENDANCE SYSTEM LAUNCHER
This file launches the unified backend app
"""

import sys
import os

# Add backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

print("🔄 Launching BHK Tech Attendance System...")
print(f"📁 Backend path: {backend_path}")

# Change to backend directory and run the app
os.chdir(backend_path)

if __name__ == '__main__':
    print("🚀 Starting from main_app.py launcher...")
    
    # Import and run the main app
    try:
        from app import app, socketio
        
        # Start the server directly
        print("✅ Imported unified backend app")
        
        # Run the app
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # Avoid conflicts with SocketIO
        )
        
    except ImportError as e:
        print(f"❌ Failed to import backend app: {e}")
        print("💡 Try running the backend directly: cd backend && python app.py")
    except Exception as e:
        print(f"❌ Error starting application: {e}")