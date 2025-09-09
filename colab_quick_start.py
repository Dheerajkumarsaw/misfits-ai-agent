"""
Quick Google Colab Setup Script
Copy and paste this into Google Colab cells
"""

# ===== CELL 1: Install Packages =====
"""
!pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 pydantic==2.5.0
!pip install chromadb==0.4.18 openai==1.3.5 pandas==2.1.3 numpy==1.26.2
!pip install sentence-transformers==2.2.2 requests==2.31.0 python-multipart==0.0.6
!pip install pyngrok nest-asyncio

print("✅ All packages installed!")
"""

# ===== CELL 2: Setup ngrok =====
"""
from pyngrok import ngrok
import nest_asyncio

# Allow nested event loops (required for Colab)
nest_asyncio.apply()

# Optional: Add your ngrok auth token for better limits
# ngrok.set_auth_token("your_token_here")

print("🔧 ngrok setup completed!")
"""

# ===== CELL 3: Upload Files =====
"""
from google.colab import files

print("📁 Upload your files (live-chorma.py and api_server.py):")
uploaded = files.upload()

print("✅ Files uploaded:", list(uploaded.keys()))
"""

# ===== CELL 4: Start Server =====
"""
import subprocess
import threading
import time
from pyngrok import ngrok

def start_server():
    subprocess.run([
        "python", "-m", "uvicorn", 
        "api_server:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ])

# Start server in background
print("🚀 Starting server...")
server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

# Wait for server to start
time.sleep(5)

# Create public URL
public_url = ngrok.connect(8000)
print(f"🌐 Your API is live at: {public_url}")
print(f"📱 Main endpoint: {public_url}/api/recommendations")
"""

# ===== CELL 5: Test API =====
"""
import requests
import json

# Update with your ngrok URL
API_URL = "https://your-ngrok-id.ngrok.io"

# Test request
test_data = {
    "query": "cricket events this weekend",
    "user_current_city": "noida",
    "limit": 3
}

response = requests.post(f"{API_URL}/api/recommendations", json=test_data)
print("📊 Response:", response.json())
"""

# ===== CELL 6: Keep Running =====
"""
print("🔄 Server is running...")
print("💡 Keep this cell running to maintain the API")

try:
    while True:
        tunnels = ngrok.get_tunnels()
        if tunnels:
            print(f"🌐 Active: {tunnels[0].public_url}")
        time.sleep(30)
except KeyboardInterrupt:
    ngrok.kill()
    print("🛑 Stopped")
"""