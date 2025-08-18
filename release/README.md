# Solar Sage — Clean/Dirty Classifier (Local Demo)

This repo gives you:

- **FastAPI** server with endpoints:
  - `GET /health` – quick check
  - `POST /predict-image` – single image
  - `POST /predict-batch` – multiple images
- **Browser clients** (static):
  - `client.html` – upload one image
  - `batch.html` – upload many and download CSV
- **CLI tool**:
  - `tools/predict_folder.py` – predict a whole folder and save CSV
- **Optional saving** of uploads to `uploads/clean` or `uploads/dirty`

---

## 1) Setup (first time)

```bash
cd ~/Desktop/Solar_Sage_AI
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" numpy requests python-multipart

## Final model settings (chosen after sweep)
- THRESH: **0.40** (Acc 0.879 / Prec 0.873 / Rec 0.833 / F1 0.852 on our val split)
- Labels: ["CLEAN", "DIRTY"] (see `service/labels.json`)
- Normalization: IMAGENET (in both training & inference)

