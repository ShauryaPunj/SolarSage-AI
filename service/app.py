from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Tuple, Dict, Any, List
import numpy as np
import time, os, io, sys

# lazy heavy deps (only if needed)
_ort = None   # onnxruntime
_cv2 = None   # opencv

# ======== CONFIG TOGGLES ========
CLASS_ORDER = ("CLEAN", "DIRTY")   # flip to ("DIRTY","CLEAN") if reversed
COLOR_SPACE = "RGB"                # "RGB" (default) or "BGR"
NORM = "IMAGENET"
THRESH      = 0.40                 # decision threshold on p_dirty
DEBUG_RAW   = True                 # print debug to terminal (truncated)
HEURISTIC_IF_1000 = True           # use heuristic if model has 1000 classes

# saving uploads
SAVE_UPLOADS = True
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
# =================================

app = FastAPI(title="Solar Panel Clean/Dirty Inference API", version="1.0.0")

# CORS for local dev (open)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Model init ----------
APP_DIR     = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(APP_DIR, "model.onnx")

SESS        = None
PROVIDER    = "CPUExecutionProvider"
IN_NAME     = None
IN_SHAPE    = None
OUT_SHAPE   = None
NCHW        = True                 # inferred from model input shape
MODEL_CLASSES = None               # e.g., 2 or 1000

def _lazy_import_heavy():
    global _ort, _cv2
    if _ort is None:
        try:
            import onnxruntime as ort  # type: ignore
            _ort = ort
        except Exception:
            _ort = None
    if _cv2 is None:
        try:
            import cv2  # type: ignore
            _cv2 = cv2
        except Exception:
            _cv2 = None

def _try_load_model():
    """
    Attempt to load model.onnx if present. If not present or any error, stay
    in heuristic mode.
    """
    global SESS, IN_NAME, IN_SHAPE, OUT_SHAPE, NCHW, MODEL_CLASSES
    if not os.path.exists(MODEL_PATH):
        SESS = None
        MODEL_CLASSES = None
        return

    _lazy_import_heavy()
    if _ort is None:
        # onnxruntime not installed; fallback to heuristic
        SESS = None
        MODEL_CLASSES = None
        return

    try:
        SESS = _ort.InferenceSession(MODEL_PATH, providers=[PROVIDER])
        inp = SESS.get_inputs()[0]
        out = SESS.get_outputs()[0]
        IN_NAME  = inp.name
        IN_SHAPE = inp.shape
        OUT_SHAPE = out.shape

        # infer channels-first vs channels-last
        # common: [1,3,H,W] or [N,3,H,W]  vs  [1,H,W,3] or [N,H,W,3]
        if len(IN_SHAPE) == 4:
            # find where channel dim (3) sits
            if IN_SHAPE[1] == 3:
                NCHW = True
            elif IN_SHAPE[-1] == 3:
                NCHW = False
            else:
                # default
                NCHW = True
        else:
            NCHW = True

        # classes from output shape (last dim)
        if isinstance(OUT_SHAPE, (list, tuple)) and len(OUT_SHAPE) >= 1:
            MODEL_CLASSES = OUT_SHAPE[-1] if isinstance(OUT_SHAPE[-1], int) else None
        else:
            MODEL_CLASSES = None

    except Exception as e:
        print("[WARN] failed to load model.onnx:", e, file=sys.stderr)
        SESS = None
        MODEL_CLASSES = None

_try_load_model()

# --------- small helpers ---------
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.max()
    e = np.exp(x)
    s = e.sum()
    return e / (s if s != 0 else 1.0)

def _img_bytes_to_gray(raw: bytes) -> np.ndarray:
    """
    Decode image to grayscale using cv2 if available; otherwise fallback to
    byte-level approximation (not ideal, but keeps server running).
    """
    _lazy_import_heavy()
    if _cv2 is not None:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = _cv2.imdecode(arr, _cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    # fallback: fake "gray" from raw bytes reshaped
    b = np.frombuffer(raw, dtype=np.uint8)
    if b.size < 4096:
        # pad so stats aren't too wild for very tiny files
        pad = np.zeros(4096 - b.size, dtype=np.uint8)
        b = np.concatenate([b, pad], axis=0)
    side = int(np.sqrt(b.size))
    b = b[: side * side]
    return b.reshape(side, side)

def _heuristic_predict(raw: bytes) -> Tuple[str, float, Dict[str, Any]]:
    """
    Very simple heuristic: compute a 'dirtiness' proxy from contrast + edges if cv2 is present,
    otherwise from gray-level stddev. Returns (label, score, meta).
    """
    _lazy_import_heavy()
    gray = _img_bytes_to_gray(raw)

    if _cv2 is not None:
        # edge density via Canny
        try:
            edges = _cv2.Canny(gray, 50, 150)
            edge_ratio = (edges > 0).mean()
        except Exception:
            edge_ratio = 0.1
        # brightness/contrast
        mean = float(gray.mean())
        std = float(gray.std() + 1e-6)

        # crude p_dirty from edges + low mean (darker) + high std
        p_dirty = 0.5 * edge_ratio + 0.25 * (1.0 - (mean / 255.0)) + 0.25 * min(std / 128.0, 1.0)
    else:
        # no cv2: use std + inverted mean on fake gray
        mean = float(gray.mean())
        std = float(gray.std() + 1e-6)
        p_dirty = 0.6 * (1.0 - (mean / 255.0)) + 0.4 * min(std / 128.0, 1.0)

    p_dirty = float(np.clip(p_dirty, 0.0, 1.0))
    label = CLASS_ORDER[1] if p_dirty >= THRESH else CLASS_ORDER[0]

    meta = {"mode": "heuristic", "mean": mean, "std": std}
    return label, p_dirty, meta

def _prep_for_model(raw: bytes) -> np.ndarray:
    """
    Decode + preprocess to match the model input.
    """
    _lazy_import_heavy()
    if _cv2 is None:
        raise RuntimeError("opencv-python is required to use the ONNX model")

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # convert BGR->RGB if requested
    if COLOR_SPACE.upper() == "RGB":
        img = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)

    # find target H,W
    if IN_SHAPE is None or len(IN_SHAPE) != 4:
        # default to 224x224
        H, W = 224, 224
    else:
        if NCHW:
            H, W = IN_SHAPE[2], IN_SHAPE[3]
        else:
            H, W = IN_SHAPE[1], IN_SHAPE[2]
        # handle None dims (dynamic)
        H = 224 if (H is None or isinstance(H, str)) else int(H)
        W = 224 if (W is None or isinstance(W, str)) else int(W)

    img = _cv2.resize(img, (W, H), interpolation=_cv2.INTER_LINEAR)
    x = img.astype(np.float32)

    # normalize
    if NORM == "0_1":
        x /= 255.0
    elif NORM == "NEG1_1":
        x = (x / 127.5) - 1.0
    elif NORM == "IMAGENET":
        # imagenet means RGB mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x / 255.0 - mean) / std

    # to NCHW or NHWC
    if NCHW:
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
        x = np.expand_dims(x, 0)        # -> 1xCxHxW
    else:
        x = np.expand_dims(x, 0)        # -> 1xHxWxC

    return x

def _model_predict(raw: bytes) -> Tuple[str, float, Dict[str, Any]]:
    """
    Run ONNX model if loaded. Supports 2-class directly; if 1000-class and
    HEURISTIC_IF_1000 is True, use heuristic instead.
    """
    if SESS is None:
        raise RuntimeError("Model session not available")

    # 1000-class imageNet style => use heuristic unless disabled
    if MODEL_CLASSES == 1000 and HEURISTIC_IF_1000:
        label, p_dirty, meta = _heuristic_predict(raw)
        meta["note"] = "ImageNet-style model detected; using heuristic fallback"
        return label, p_dirty, meta

    x = _prep_for_model(raw)
    inputs = {SESS.get_inputs()[0].name: x}
    y = SESS.run(None, inputs)[0]

    # y shape: [1, C]
    logits = y[0].astype(np.float32)
    probs = _softmax(logits)

    # map to CLEAN/DIRTY
    # assume order of probs is [CLEAN, DIRTY] if 2-class
    if MODEL_CLASSES == 2 and probs.shape[-1] == 2:
        p_clean, p_dirty = float(probs[0]), float(probs[1])
    else:
        # unknown head: use heuristic as safest, but keep probs for debug
        label_h, p_dirty_h, meta_h = _heuristic_predict(raw)
        if DEBUG_RAW:
            print("--- PRED DEBUG (unknown-head) ---")
            print("logits (trunc):", np.array2string(logits[:16], precision=3), flush=True)
            print("probs  (trunc):", np.array2string(probs[:16],  precision=3), flush=True)
        meta_h["note"] = "Unknown model head; heuristic used"
        return label_h, p_dirty_h, meta_h

    label = CLASS_ORDER[1] if p_dirty >= THRESH else CLASS_ORDER[0]

    if DEBUG_RAW:
        print("--- PRED DEBUG ---")
        print("logits (trunc):", np.array2string(logits[:16], precision=3), flush=True)
        print(f"p_clean={p_clean:.3f} p_dirty={p_dirty:.3f}  THRESH={THRESH}")
        print("mode:", "model", "label:", label, f"score: {p_dirty:.3f}", flush=True)

    meta = {"mode": "model"}
    return label, p_dirty, meta

def _classify_bytes(raw: bytes, filename: str = "upload.jpg") -> Dict[str, Any]:
    """
    Core classifier used by single + batch endpoints.
    """
    backend = "heuristic"
    if SESS is not None:
        try:
            label, p_dirty, meta = _model_predict(raw)
            backend = meta.get("mode", "model")
        except Exception as e:
            # fall back to heuristic if model path fails for any reason
            print("[WARN] model path failed; using heuristic:", e, file=sys.stderr)
            label, p_dirty, meta = _heuristic_predict(raw)
            backend = meta.get("mode", "heuristic")
    else:
        label, p_dirty, meta = _heuristic_predict(raw)
        backend = meta.get("mode", "heuristic")

    # score is p_dirty for consistency
    result = {
        "label": label,
        "score": float(p_dirty),
        "meta": {"latency_ms": None, "mode": backend}
    }
    return result

# ------------- routes -------------
@app.get("/health")
def health():
    backend = "model" if SESS is not None and MODEL_CLASSES != 1000 else "heuristic"
    return {"status": "ok", "model": "v1.0", "backend": backend}

@app.get("/model-info")
def model_info():
    return {
        "provider": PROVIDER,
        "input_shape": IN_SHAPE,
        "output_shape": OUT_SHAPE,
        "channels_first": NCHW,
        "model_loaded": SESS is not None,
        "model_classes": MODEL_CLASSES,
        "class_order": CLASS_ORDER,
        "norm": NORM,
        "color_space": COLOR_SPACE,
        "heuristic_if_1000": HEURISTIC_IF_1000,
    }

@app.post("/predict-image")
async def predict_image(image: UploadFile = File(...)):
    # basic type check
    ctype = (image.content_type or "").lower()
    if not (ctype.endswith("jpeg") or ctype.endswith("jpg") or ctype.endswith("png")):
        raise HTTPException(status_code=415, detail="Only JPEG/PNG allowed")

    t0 = time.time()
    raw = await image.read()
    result = _classify_bytes(raw, filename=image.filename)
    result["meta"]["latency_ms"] = int((time.time() - t0) * 1000)

    # optional save
    if SAVE_UPLOADS:
        try:
            sub = str(result["label"]).lower()
            os.makedirs(os.path.join(UPLOAD_DIR, sub), exist_ok=True)
            fname = f"{int(time.time()*1000)}_{os.path.basename(image.filename or 'upload.jpg')}"
            with open(os.path.join(UPLOAD_DIR, sub, fname), "wb") as f:
                f.write(raw)
        except Exception as e:
            print("[WARN] failed to save upload:", e)

    return result

# ========== Batch prediction ==========
@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Accepts multiple images under the same form field name: files
    curl example:
      curl -s -X POST \
        -F "files=@samples/001.png" \
        -F "files=@samples/002.png" \
        http://127.0.0.1:8000/predict-batch
    """
    out: List[Dict[str, Any]] = []
    for f in files:
        ctype = (f.content_type or "").lower()
        if not (ctype.endswith("jpeg") or ctype.endswith("jpg") or ctype.endswith("png")):
            raise HTTPException(status_code=415, detail="Only JPEG/PNG allowed in batch")

        raw = await f.read()
        t0 = time.time()
        result = _classify_bytes(raw, filename=f.filename)
        result["meta"]["latency_ms"] = int((time.time() - t0) * 1000)

        # optional save per-file
        if SAVE_UPLOADS:
            try:
                sub = str(result["label"]).lower()
                os.makedirs(os.path.join(UPLOAD_DIR, sub), exist_ok=True)
                fname = f"{int(time.time()*1000)}_{os.path.basename(f.filename or 'upload.jpg')}"
                with open(os.path.join(UPLOAD_DIR, sub, fname), "wb") as w:
                    w.write(raw)
            except Exception as e:
                print("[WARN] failed to save upload in batch:", e)

        item = dict(result)
        item["file"] = os.path.basename(f.filename or f"file{len(out)}.jpg")
        out.append(item)

    return out
# ======== end batch =========

