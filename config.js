// Configuration file for TMT Ring Test App
import { Platform } from 'react-native';

// Deployed backend URL (for web/production)
const DEPLOYED_BACKEND_URL = 'https://unwonted-uplift-tmt-analyzer-backend.hf.space';

// Local backend URL (for mobile/development)
// IMPORTANT: Update this with your laptop's IP address
// To find your IP:
// - Windows: Open Command Prompt and run 'ipconfig'
// - Look for "IPv4 Address" under your WiFi adapter
// - Example: 192.168.1.100
const LOCAL_BACKEND_URL = 'http://192.168.1.19:5000';

// Automatically use deployed URL on web, local URL on mobile
export const BACKEND_URL = Platform.OS === 'web' 
  ? DEPLOYED_BACKEND_URL 
  : LOCAL_BACKEND_URL;

// Make sure:
// 1. Your phone and laptop are on the same WiFi network (for mobile)
// 2. The backend is running (python app.py) (for mobile)
// 3. Windows Firewall allows Python through (for mobile)
// 4. Port 5000 is not blocked (for mobile)
// 5. For web, the deployed backend at https://unwonted-uplift-tmt-analyzer-backend.hf.space is used

export default {
  BACKEND_URL,
  DEPLOYED_BACKEND_URL,
  LOCAL_BACKEND_URL,
}; 