#!/usr/bin/env python3
"""
Quick test to verify backend is working
"""

import requests
import json

def test_backend():
    """Test if backend is responding correctly"""
    try:
        # Test the home endpoint
        response = requests.get('http://192.168.1.35:5000/')
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is working!")
            print(f"   Status: {data.get('status')}")
            print(f"   SAM Model: {data.get('sam_model_loaded')}")
            print(f"   Endpoints: {data.get('endpoints')}")
            return True
        else:
            print(f"❌ Backend returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return False

if __name__ == "__main__":
    print("Testing backend connection...")
    test_backend() 