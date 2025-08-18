from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, time, random

app = FastAPI(title="Solar Panel Clean/Dirty Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "model": "v1.0"}

def infer_mock(image_bytes: bytes):
    time.sleep(0.03)  # simulate latency
    try:
        Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=415,
            detail={"error": {"code": "UNSUPPORTED_MEDIA_TYPE", "message": "Only JPEG/PNG allowed"}}
        )
    label = random.choice(["CLEAN", "DIRTY"])
    score = round(random.uniform(0.6, 0.98), 3)
    return label, score

@app.post("/predict-image")
async def predict_image(image: UploadFile = File(...)):
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=415,
            detail={"error": {"code": "UNSUPPORTED_MEDIA_TYPE", "message": "Only JPEG/PNG allowed"}}
        )
    t0 = time.time()
    image_bytes = await image.read()
    label, score = infer_mock(image_bytes)
    latency_ms = int((time.time() - t0) * 1000)
    return {"label": label, "score": float(score), "meta": {"latency_ms": latency_ms, "model_ver": "v1.0-mock"}}
