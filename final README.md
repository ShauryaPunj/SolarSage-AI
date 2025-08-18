Fully local desktop app that classifies solar-panel images as **CLEAN** or **DIRTY**.

- **Backend**: FastAPI + ONNX (ResNet18 2-class)
- **Frontend**: Electron (webcam, single upload, batch, session results, CSV export)
- **Privacy**: runs on your Mac; no cloud calls.

---

## ✨ Features

- **Live**: Use your webcam → capture → predict (label, score, latency).
- **Upload**: Select one image (JPEG/PNG) → predict.
- **Batch**: Select multiple images → predict all → per-file table → **Download Batch CSV**.
- **Results**: Full session history (Live/Upload/Batch) → **Download Session CSV**.
- **Analytics**: Totals + CLEAN / DIRTY counts.
- **Health**: Footer button checks API `/health`.

**API endpoints (unchanged)**
- `GET /health` → `{"status":"ok","model":"v1.0","backend":"model"}`
- `POST /predict-image` (field **image**)
- `POST /predict-batch` (field **files**)
- `GET /model-info`

---

## 🚀 Quick Start (macOS, Apple Silicon)

```bash
git clone <YOUR_REPO_URL> && cd Solar_Sage_AI
npm install                    # optional; launcher can fall back to npx
chmod +x start_mac.command
./start_mac.command