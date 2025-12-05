// Configuration file for TMT Ring Test App

// IMPORTANT: Update this with your laptop's IP address
// To find your IP:
// - Windows: Open Command Prompt and run 'ipconfig'
// - Look for "IPv4 Address" under your WiFi adapter
// - Example: 192.168.1.100

export const BACKEND_URL = 'http://192.168.1.19:5000';

// Make sure:
// 1. Your phone and laptop are on the same WiFi network
// 2. The backend is running (python app.py)
// 3. Windows Firewall allows Python through
// 4. Port 5000 is not blocked

export default {
  BACKEND_URL,
}; 