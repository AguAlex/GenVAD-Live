from __future__ import annotations

import base64
import os
import sys

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Asigură-te că rulezi uvicorn din folderul unde stau aceste fișiere + pachetele model/util copiate
from image_to_batch import image_bytes_to_sample_batch
from inference_config import CONFIG
from model.model_factory import mae_cvt_patch16, mae_cvt_patch8

MC_PREDICT_RUNS = 30

app = FastAPI(title="GenVAD v2 predict")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: torch.nn.Module | None = None
_device: torch.device | None = None


def _patch_error_map_to_png_base64(err_hw: np.ndarray, out_h: int, out_w: int) -> str:
    """Normalizează eroarea pe patch (h_p×w_p), colormap INFERNO, redimensionează la intrarea modelului."""
    vmin = float(err_hw.min())
    vmax = float(err_hw.max())
    if vmax - vmin < 1e-12:
        u8 = np.zeros(err_hw.shape, dtype=np.uint8)
    else:
        u8 = ((err_hw - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    color = cv2.resize(color, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".png", color)
    if not ok:
        raise RuntimeError("cv2.imencode a eșuat pentru heatmap.")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def _build_model() -> torch.nn.Module:
    if CONFIG.dataset == "avenue":
        return mae_cvt_patch16(
            img_size=CONFIG.input_size,
            use_only_masked_tokens_ab=CONFIG.use_only_masked_tokens_ab,
            masking_method=CONFIG.masking_method,
        ).float()
    return mae_cvt_patch8(
        img_size=CONFIG.input_size,
        use_only_masked_tokens_ab=CONFIG.use_only_masked_tokens_ab,
        masking_method=CONFIG.masking_method,
    ).float()


def get_model() -> torch.nn.Module:
    global _model, _device
    if _model is not None:
        return _model

    ckpt_path = CONFIG.checkpoint_path
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Lipsește checkpoint-ul: {os.path.abspath(ckpt_path)}"
        )

    _device = torch.device(
        CONFIG.device if torch.cuda.is_available() or CONFIG.device == "cpu" else "cpu"
    )
    if CONFIG.device == "cuda" and not torch.cuda.is_available():
        _device = torch.device("cpu")

    model = _build_model().to(_device)

    # compat: unele versiuni PyTorch cer weights_only=False pentru pickle complet
    try:
        checkpoint = torch.load(ckpt_path, map_location=_device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=_device)

    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=False)
    model.eval()

    _model = model
    return _model


@app.on_event("startup")
def startup_event() -> None:
    get_model()


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Trimite un fișier image/* (png, jpeg, …).")

    raw = await file.read()
    try:
        samples = image_bytes_to_sample_batch(raw, CONFIG)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    model = get_model()
    assert _device is not None
    samples = samples.to(_device, non_blocking=True)

    with torch.no_grad():
        _, _, _, recon_error = model(samples, mask_ratio=CONFIG.mask_ratio)

    scores = recon_error.detach().float().cpu().numpy().reshape(-1).tolist()

    return {
        "filename": file.filename,
        "anomaly_scores": scores,
        "anomaly_score": scores[0] if scores else None,
        "input_size_hw": list(CONFIG.input_size),
        "mask_ratio": CONFIG.mask_ratio,
        "device": str(_device),
    }


@app.post("/predict_mean")
async def predict_mean(file: UploadFile = File(...)) -> dict:
    """
    Aceeași imagine, MC_PREDICT_RUNS forward-uri (mască aleatoare diferită);
    întoarce media scorului, deviația standard, scoruri pe rulare și o hartă PNG (base64)
    a erorii medii de reconstrucție pe patch (nu este masca MAE, ci unde reconstrucția diferă mai mult).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Trimite un fișier image/* (png, jpeg, …).")

    raw = await file.read()
    try:
        samples = image_bytes_to_sample_batch(raw, CONFIG)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    model = get_model()
    assert _device is not None
    samples = samples.to(_device, non_blocking=True)

    runs: list[torch.Tensor] = []
    patch_maps: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(MC_PREDICT_RUNS):
            _loss, pred, _mask, recon_error = model(samples, mask_ratio=CONFIG.mask_ratio)
            runs.append(recon_error.detach().float())
            patch_maps.append(model.per_patch_squared_error(samples, pred))

    stacked = torch.stack(runs, dim=0)
    mean_err = stacked.mean(dim=0)
    std_err = stacked.std(dim=0, correction=0)

    mean_patch = torch.stack(patch_maps, dim=0).mean(dim=0)
    map0 = mean_patch[0].detach().float().cpu().numpy()
    H, W = CONFIG.input_size
    heatmap_b64 = _patch_error_map_to_png_base64(map0, H, W)
    h_p, w_p = map0.shape
    p = int(model.patch_embed.patch_size[0])

    scores = mean_err.cpu().numpy().reshape(-1).tolist()
    stds = std_err.cpu().numpy().reshape(-1).tolist()
    per_run = stacked[:, 0].cpu().numpy().reshape(-1).tolist()

    return {
        "filename": file.filename,
        "monte_carlo_runs": MC_PREDICT_RUNS,
        "anomaly_scores": scores,
        "anomaly_score": scores[0] if scores else None,
        "anomaly_score_std": stds[0] if stds else None,
        "per_run_scores": per_run,
        "anomaly_heatmap_png_base64": heatmap_b64,
        "heatmap_patch_grid_hw": [h_p, w_p],
        "heatmap_patch_size": p,
        "heatmap_note": (
            "Eroare medie de reconstrucție pe patch (media peste rulări), "
            "normalizată min–max pe imagine pentru afișare; nu e masca MAE aleatoare dintr-o singură rulare."
        ),
        "input_size_hw": list(CONFIG.input_size),
        "mask_ratio": CONFIG.mask_ratio,
        "device": str(_device),
    }