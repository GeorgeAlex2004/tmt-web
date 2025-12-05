# TMT BAR ANALYZER

A comprehensive mobile application for TMT (Thermo-Mechanically Treated) bar quality analysis using advanced computer vision and machine learning techniques.

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Frontend Setup (React Native/Expo)](#frontend-setup-react-nativeexpo)
  - [Backend Setup (Python Flask)](#backend-setup-python-flask)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Image Analysis**: Capture and analyze TMT bar cross-sections using camera or image gallery
- **AI-Powered Segmentation**: Uses Segment Anything Model (SAM) for precise image segmentation
- **Quality Metrics**: Calculate various quality parameters including diameter, thickness, and geometric properties
- **Report Generation**: Generate detailed PDF reports with analysis results
- **History Tracking**: View and manage analysis history
- **Multi-platform Support**: Works on iOS, Android, and Web

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

### For Frontend:
- **Node.js** (v16 or higher) - [Download](https://nodejs.org/)
- **npm** or **yarn** package manager
- **Expo CLI** (will be installed globally)
- **Expo Go** app on your mobile device (for testing)

### For Backend:
- **Python** (v3.8 or higher) - [Download](https://www.python.org/downloads/)
- **pip** (Python package manager)
- **Git** (for cloning the repository)

### Additional Requirements:
- **SAM Model Checkpoint**: Download the SAM ViT-H model checkpoint
  - Download from: [Segment Anything Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)
  - File needed: `sam_vit_h_4b8939.pth`
  - Place it in the `backend/` directory

## 📦 Installation

### Frontend Setup (React Native/Expo)

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/GeorgeAlex2004/TMT-BAR-ANALYZER.git
   cd TMT-BAR-ANALYZER
   ```

2. **Install dependencies**:
   ```bash
   # Using npm
   npm install
   
   # OR using yarn (recommended)
   yarn install
   ```

3. **Install Expo CLI globally** (if not already installed):
   ```bash
   npm install -g expo-cli
   # OR
   yarn global add expo-cli
   ```

### Backend Setup (Python Flask)

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create a Python virtual environment**:
   ```bash
   # Windows
   python -m venv tata
   
   # macOS/Linux
   python3 -m venv tata
   ```

3. **Activate the virtual environment**:
   ```bash
   # Windows (PowerShell)
   .\tata\Scripts\Activate.ps1
   
   # Windows (Command Prompt)
   .\tata\Scripts\activate.bat
   
   # macOS/Linux
   source tata/bin/activate
   ```

4. **Install Python dependencies**:
   ```bash
   # Using requirements.txt (recommended)
   pip install -r requirements.txt
   
   # OR install manually
   pip install flask flask-cors pillow numpy torch torchvision opencv-python segment-anything ultralytics scipy matplotlib psutil requests
   ```

   **Note**: If you encounter issues installing `segment-anything`, you may need to install it from source:
   ```bash
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```
   
   **Note**: PyTorch installation may require specific versions for your system. Visit [PyTorch Installation](https://pytorch.org/get-started/locally/) for system-specific installation commands.

5. **Download and place the SAM model checkpoint**:
   - Download `sam_vit_h_4b8939.pth` from the [official repository](https://github.com/facebookresearch/segment-anything#model-checkpoints)
   - Place the file in the `backend/` directory
   - The file should be at: `backend/sam_vit_h_4b8939.pth`

## ⚙️ Configuration

### Backend URL Configuration

1. **Find your computer's IP address**:
   - **Windows**: Open Command Prompt and run `ipconfig`
     - Look for "IPv4 Address" under your WiFi adapter
   - **macOS/Linux**: Run `ifconfig` or `ip addr`
     - Look for your network interface IP address

2. **Update the frontend configuration**:
   - Open `config.js` in the root directory
   - Update the `BACKEND_URL` with your IP address:
     ```javascript
     export const BACKEND_URL = 'http://YOUR_IP_ADDRESS:5000';
     ```
   - Example: `http://192.168.1.100:5000`

3. **Important Notes**:
   - Your phone/device and computer must be on the **same WiFi network**
   - Ensure Windows Firewall allows Python through (or temporarily disable it for testing)
   - Port 5000 should not be blocked by your firewall

## 🚀 Running the Application

### Step 1: Start the Backend Server

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Activate the virtual environment** (if not already activated):
   ```bash
   # Windows (PowerShell)
   .\tata\Scripts\Activate.ps1
   
   # Windows (Command Prompt)
   .\tata\Scripts\activate.bat
   
   # macOS/Linux
   source tata/bin/activate
   ```

3. **Start the Flask server**:
   ```bash
   python app.py
   # OR
   python unified_backend.py
   ```

4. **Verify the server is running**:
   - You should see output indicating the server is running on `http://0.0.0.0:5000` or `http://127.0.0.1:5000`
   - The SAM model should load successfully (this may take a few moments)

### Step 2: Start the Frontend Application

1. **Open a new terminal** and navigate to the project root:
   ```bash
   cd TMT-BAR-ANALYZER
   ```

2. **Start the Expo development server**:
   ```bash
   npm start
   # OR
   yarn start
   # OR
   expo start
   ```

3. **Run on your preferred platform**:
   - **Mobile Device**: 
     - Install Expo Go app from App Store (iOS) or Play Store (Android)
     - Scan the QR code displayed in the terminal
   - **Android Emulator**: 
     ```bash
     npm run android
     # OR
     yarn android
     ```
   - **iOS Simulator** (macOS only):
     ```bash
     npm run ios
     # OR
     yarn ios
     ```
   - **Web Browser**:
     ```bash
     npm run web
     # OR
     yarn web
     ```

## 📁 Project Structure

```
TMT-BAR-ANALYZER/
├── assets/                 # Images, icons, and static assets
├── backend/               # Python Flask backend
│   ├── app.py            # Main Flask application
│   ├── unified_backend.py # Unified backend implementation
│   ├── analysis.py       # Image analysis functions
│   ├── Singular/         # Singular analysis modules
│   │   ├── angle.py
│   │   ├── distance.py
│   │   ├── height.py
│   │   └── length.py
│   ├── sam_vit_h_4b8939.pth  # SAM model checkpoint (download separately)
│   └── tata/             # Python virtual environment (gitignored)
├── components/           # Reusable React components
├── screens/              # Application screens
├── src/                  # TypeScript source files
│   ├── components/      # TypeScript components
│   ├── screens/         # TypeScript screens
│   ├── services/        # API and service files
│   └── models/          # Data models
├── utils/                # Utility functions
├── config.js             # Backend URL configuration
├── App.js                # Main application entry point
├── package.json          # Node.js dependencies
└── README.md            # This file
```

## 🔍 Troubleshooting

### Backend Issues

**Problem**: SAM model not loading
- **Solution**: Ensure `sam_vit_h_4b8939.pth` is in the `backend/` directory
- Verify the file is not corrupted and is the correct model checkpoint

**Problem**: Port 5000 already in use
- **Solution**: 
  - Change the port in `app.py` or `unified_backend.py`
  - Update `BACKEND_URL` in `config.js` accordingly

**Problem**: Module not found errors
- **Solution**: 
  - Ensure virtual environment is activated
  - Reinstall dependencies: `pip install -r requirements.txt` (if available) or install packages individually

**Problem**: CORS errors
- **Solution**: 
  - Verify `flask-cors` is installed
  - Check that `CORS(app)` is enabled in the Flask app

### Frontend Issues

**Problem**: Cannot connect to backend
- **Solution**:
  - Verify backend is running on the correct port
  - Check that `BACKEND_URL` in `config.js` matches your computer's IP
  - Ensure both devices are on the same WiFi network
  - Check firewall settings

**Problem**: Expo start fails
- **Solution**:
  - Clear cache: `expo start -c` or `npm start -- --clear`
  - Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`

**Problem**: Metro bundler issues
- **Solution**:
  - Reset Metro cache: `npx react-native start --reset-cache`
  - Clear watchman: `watchman watch-del-all` (if using watchman)

### General Issues

**Problem**: Dependencies installation fails
- **Solution**:
  - Use `yarn` instead of `npm` (or vice versa)
  - Clear package manager cache
  - Ensure Node.js version is compatible (v16+)

**Problem**: Python virtual environment issues
- **Solution**:
  - Recreate the virtual environment
  - Ensure Python 3.8+ is installed
  - Use `python -m venv` instead of `virtualenv`

## 📝 Notes

- The SAM model file (`sam_vit_h_4b8939.pth`) is large (~2.4GB) and is not included in the repository. You must download it separately.
- First-time model loading may take several minutes depending on your hardware.
- For production use, consider using a more robust backend deployment (e.g., Gunicorn, Docker).
- Ensure sufficient disk space for Python packages and model files.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is private and proprietary.

## 👥 Authors

- **GeorgeAlex2004** - [GitHub](https://github.com/GeorgeAlex2004)

## 🙏 Acknowledgments

- Facebook Research for the Segment Anything Model (SAM)
- Expo team for the excellent React Native framework
- All open-source contributors whose libraries made this project possible

