from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, onnxruntime as ort, cv2, time, os

# ======== CONFIG TOGGLES (flip & test) ========
CLASS_ORDER = ("CLEAN", "DIRTY")   # if predictions look flipped, set to ("DIRTY","CLEAN")
COLOR_SPACE = "RGB"                 # try "BGR" if RGB doesn't work
NORM        = "0_1"                 # try "IMAGENET" or "NEG1_1" if still off
THRESH      = 0.5                   # threshold for binary (1-logit) models
DEBUG_RAW   = True                  # prints logits/probs in terminal
# =============================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.onnx")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"model.onnx not found at {MODEL_PATH}")

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
IN_NAME  = inp.name
IN_SHAPE = inp.shape
providers = sess.get_providers()

def _get_hw_nchw():
    H = IN_SHAPE[2] if isinstance(IN_SHAPE[2], int) else 224
    W = IN_SHAPE[3] if isinstance(IN_SHAPE[3], int) else 224
    return H, W

def _get_hw_nhwc():
    H = IN_SHAPE[1] if isinstance(IN_SHAPE[1], int) else 224
    W = IN_SHAPE[2] if isinstance(IN_SHAPE[2], int) else 224
    return H, W

def _normalize(x):
    if NORM == "0_1":
        return x.astype(np.float32) / 255.0
    if NORM == "NEG1_1":
        return (x.astype(np.float32) / 255.0 - 0.5) / 0.5
    if NORM == "IMAGENET":
        x = x.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (x - mean) / std
    return x.astype(np.float32) / 255.0

def preprocess_bgr(img_bgr):
    nchw = (len(IN_SHAPE) == 4 and isinstance(IN_SHAPE[1], int) and IN_SHAPE[1] == 3)
    if nchw:
        H, W = _get_hw_nchw()
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if COLOR_SPACE == "RGB" else img_bgr.copy()
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        x = _normalize(img)
        x = np.transpose(x, (2, 0, 1))[None, ...]  # [1,3,H,W]
    else:
        H, W = _get_hw_nhwc()
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if COLOR_SPACE == "RGB" else img_bgr.copy()
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        x = _normalize(img)[None, ...]              # [1,H,W,3]
    return x

@app.get("/health")
def health():
    return {"status": "ok", "model": "onnx-runtime"}

@app.get("/model-info")
def model_info():
    outs = [{"name": o.name, "shape": o.shape, "type": o.type} for o in sess.get_outputs()]
    layout = "NCHW" if (len(IN_SHAPE)==4 and isinstance(IN_SHAPE[1], int) and IN_SHAPE[1]==3) else "NHWC"
    return {
        "input": {"name": IN_NAME, "shape": IN_SHAPE, "type": inp.type},
        "outputs": outs,
        "providers": providers,
        "config": {"layout": layout, "color_space": COLOR_SPACE, "norm": NORM, "class_order": CLASS_ORDER}
    }

@app.post("/predict-image")
async def predict_image(image: UploadFile = File(...)):
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=415,
            detail={"error":{"code":"UNSUPPORTED_MEDIA_TYPE","message":"Only JPEG/PNG allowed"}}
        )
    buf = np.frombuffer(await image.read(), np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail={"error":{"code":"BAD_IMAGE","message":"Decode failed"}})

    t0 = time.time()
    x = preprocess_bgr(img_bgr)
    out = sess.run(None, {IN_NAME: x})[0]  # logits or probs

    # Normalize shape
    if out.ndim == 2 and out.shape[0] == 1:
        out = out[0]

    # 1-logit -> sigmoid; >=2 -> softmax
    if out.shape[-1] == 1:
        logit = float(out[0])
        p_dirty = 1.0 / (1.0 + np.exp(-logit))
        probs = np.array([1.0 - p_dirty, p_dirty], dtype=np.float32)  # [p_clean, p_dirty]
        idx = 1 if p_dirty >= THRESH else 0
    else:
        z = out.astype(np.float32)
        z -= z.max()
        probs = np.exp(z) / np.exp(z).sum()
        idx = int(np.argmax(probs))

    label = CLASS_ORDER[idx]
    score = float(probs[idx])

    latency_ms = int((time.time() - t0) * 1000)

    if DEBUG_RAW:
        try:
            print("\n--- PRED DEBUG ---")
            print("raw:", out.tolist() if hasattr(out, "tolist") else out)
            print("probs:", probs.tolist())
            print("label:", label, "score:", round(score, 3))
        except Exception:
            pass

    return {"label": label, "score": round(score, 3),
            "meta": {"latency_ms": latency_ms, "model_ver": "onnx"}}
