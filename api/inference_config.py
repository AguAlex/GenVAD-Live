import os
from dataclasses import dataclass

# === Editează după cum ai antrenat RUN01 ===
@dataclass(frozen=True)
class InferenceConfig:
    # "avenue" -> mae_cvt_patch16 | alt dataset din main -> patch8
    dataset: str = "avenue"

    # Trebuie să coincidă cu v2/configs/configs.py la momentul antrenării (implicit acolo e (256, 384))
    input_size: tuple[int, int] = (256, 384)  # (H, W)

    mask_ratio: float = 0.5
    masking_method: str = "random_masking"
    use_only_masked_tokens_ab: bool = False

    # Calea către checkpoint (poate fi absolută sau relativă la cwd când pornești uvicorn)
    checkpoint_path: str = os.path.join(
        "weights", "checkpoint-best.pth"
    )

    # cuda sau cpu
    device: str = "cuda"


CONFIG = InferenceConfig()