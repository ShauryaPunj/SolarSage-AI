# Solar Sage AI (Desktop)

**Solar Sage AI** is a fully local desktop application that helps you quickly check the cleanliness of solar panels using **AI image classification**.  

The app uses:
- **Backend**: FastAPI serving a ResNet18 ONNX model (CLEAN vs DIRTY)
- **Frontend**: Electron desktop interface (webcam capture + uploads + batch + results + analytics)

Runs entirely offline — no cloud required.  

---

## ✨ Features

### 🔴 Live Camera Mode
- Accesses your Mac’s webcam.
- Click **Capture** to take a photo.
- The app sends the image to the FastAPI backend → AI model predicts **CLEAN** or **DIRTY**.
- Instant feedback with probability score.

### 📂 Upload Mode
- Upload a single image file (JPEG/PNG).
- See the prediction result with label + probability.
- Great for testing sample solar panel images.

### 📦 Batch Mode
- Select an entire folder of images.
- Each image is processed automatically by the model.
- Results displayed in a table (filename + prediction + probability).
- **Download CSV** button exports results to a `.csv` file for record-keeping.

### 📊 Results Tab
- Shows your session history (all predictions you’ve made since launch).
- Includes predictions from Live, Upload, and Batch.
- Export session results as CSV for documentation.

### 📈 Analytics Tab
- Aggregates your session:
  - Total images processed
  - % CLEAN vs DIRTY
  - Bar chart of class distribution
- Useful for quick insights after scanning a batch.

### ⚡ FastAPI Backend
- Runs locally on port `8000`.
- Exposes these endpoints:
  - `GET /health` → `{"status":"ok","model":"v1.0","backend":"model"}`
  - `POST /predict-image` → classify a single uploaded image
  - `POST /predict-batch` → classify multiple files
  - `GET /model-info` → metadata about the ONNX model

### 📑 Documentation & Credits
- **MODEL_CARD.md** → model provenance, metrics, hash
- **AUTHORS.md** → contributors and roles
- **CHANGELOG.md** → version history
- **LICENSE** → legal terms (MIT/Apache)

---

## 🚀 Quick Start (macOS, Apple Silicon)

Clone and run:

```bash
git clone <YOUR_REPO_URL> && cd Solar_Sage_AI
npm install   # installs dev electron
chmod +x start_mac.command
./start_mac.command
