# SolarSage AI - Windows Setup Guide

Quick setup guide for running SolarSage AI on Windows.

## Prerequisites

1. Install [Python](https://www.python.org/downloads/) (3.8 or higher)
2. Install [Node.js](https://nodejs.org/) (Latest LTS version)
3. Install [Git](https://git-scm.com/downloads)

## Quick Start Commands

### 1. Clone and Setup
```powershell
# Clone repository
git clone https://github.com/ShauryaPunj/SolarSage-AI.git
cd SolarSage-AI

# Install frontend dependencies
npm install

# Install backend dependencies
cd service
pip install -r ..\requirements.txt
pip install python-multipart
```

### 2. Run the Application

Open two separate terminal windows:

**Terminal 1 - Backend:**
```powershell
cd service
python -m uvicorn app:app --reload
```

**Terminal 2 - Frontend:**
```powershell
# Make sure you're in the project root directory
cd SolarSage-AI
npm start
```

### 3. Verify Installation

1. The backend should be running on: http://127.0.0.1:8000
2. The Electron app should open automatically
3. Click the "Health" button in the app footer - should show OK status

### Troubleshooting

If the app doesn't start:
1. Make sure both terminals are running
2. Verify port 8000 is not in use
3. Check if all dependencies are installed correctly
4. Try running `npm install` again if frontend doesn't start
