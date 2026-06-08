"""
Generează local `train/frames_abnormal` și `train/masks_abnormal` (oglindă față de `train/frames`),
folosind Gemini (bbox + prompt) + Juggernaut XL inpainting.

Dataset-uri suportate:
  - avenue: cadre .png / .jpg / .jpeg (implicit ./Avenue_Dataset)
  - ucsd:   cadre .tif / .tiff (implicit ./UCSD_Dataset)

Exemple:
  python Scripts/generate_anomalies.py --dataset avenue --data_root ./Avenue_Dataset --num_runs 10
  python Scripts/generate_anomalies.py --dataset ucsd --data_root ./UCSD_Dataset --num_runs 5 --seed 42

  # Măști pe Drive, cadre anormale lângă dataset (Colab):
  python Scripts/generate_anomalies.py --dataset ucsd --data_root /content/UCSD_Dataset \\
    --masks_dir "/content/drive/MyDrive/Licenta/export/train/masks_abnormal" --num_runs 20

  # Doar anumite tipuri (ID-uri din ANOMALY_CATALOG, vezi comentariul mare în fișier):
  python Scripts/generate_anomalies.py --dataset ucsd --data_root ./UCSD_Dataset \\
    --anomaly_ids A1,A3,B2 --num_runs 30

  # 5 rulări VLM × 5 variante Juggernaut = 25 anomalii (implicit --inpaint_variants 5):
  python Scripts/generate_anomalies.py --dataset avenue --data_root ./Avenue_Dataset \\
    --num_runs 5 --inpaint_variants 5 --seed 42
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import sys
import time

import cv2
import google.generativeai as genai
import numpy as np
import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting

GEMINI_API_KEY = "AIzaSyCBAIiD1tOZtv9NNvzi2SLKM5XVGO8LDf4"
genai.configure(api_key=GEMINI_API_KEY)

# =============================================================================
# REPERTORIU ANOMALII — referință pentru ID-uri (alege și pui la --anomaly_ids)
# -----------------------------------------------------------------------------
# Copiază ID-urile dorite într-o listă separată prin virgulă, ex.:  A1,A5,B3,C2
#
#   A1  — persoană adultă/adolescent prăbușită pe jos (nu în picioare), CCTV
#   A2  — rucsac mare / geantă sport abandonată pe trotuar
#   A3  — bicicletă sau trotinetă electrică căzută pe sol, roți vizibile
#   A4  — con de circulație răsturnat sau bariieră plastic mică
#   A5  — grămadă gunoi / cutie carton / resturi pe marginea trotuarului
#   A6  — foc mic la sol / smoldering, fum discret (nu perete de flăcări)
#   A7  — valiză cu roți lăsată singură (verticală sau ușor înclinată)
#   A8  — skateboard lăsat plat pe asfalt
#   A9  — pată întunecată / umedă tip scurgere ulei pe beton
#   A10 — creangă copac sau obiect lung subțire pe cărare
#   A11 — umbrelă închisă abandonată pe jos
#   A12 — cască de biciclist pe sol
#   A13 — pereche de pantofi lângă marginea drumului (abandon)
#   A14 — sac menajer negru legat lăsat pe trotuar
#   A15 — palet de lemn mic / ladă pe carosabil sau trotuar
#   B1  — persoană așezată jos cu spatele la zid (fără scaun), posibil homeless
#   B2  — cărucior de cumpărături răsturnat sau blocat singur
#   B3  — animal mediu (câine) fără lesă, pe trotuar (nu lângă stăpân vizibil)
#   B4  — motocicletă / scuter căzut pe lateral
#   B5  — extinctor roșu pe jos lângă clădire
#   B6  — scară pliabilă metalică pe sol (zonă nepotrivită)
#   B7  — furtun de incendiu sau cablu gros pe trotuar
#   B8  — grătar portabil / recipient metalic pe iarbă sau beton
#   B9  — mormane de frunze uscate nefiresc de mari (anomalie de întreținere)
#   B10 — semn rutier temporar răsturnat (plastic reflectorizant)
#   B11 — pătură / folie plastic pe sol (tabără improvizată)
#   B12 — roată de rezervă auto pe trotuar
#   B13 — portbagaj auto deschis / ușă portbagaj ridicată fără mașină în cadru
#   B14 — mormânt de pământ excavat mic / groapă șantier nebaricadată
#   B15 — stivă de pavele / boltari neîngrădită pe trotuar
#   B16 — persoană pe bicicletă în mișcare (pedalează / merge cu bicicleta), CCTV
#   C1  — fum dens care iese din canalizare / capac canal (subtile)
#   C2  — reflexie puternică neobișnuită (nu obiect) — evită dacă e ambiguu
#   C3  — petă de vopsea / var proaspăt pe asfalt
#   C4  — gaură mare în asfalt / denivelare bruscă
#   C5  — baltă stagnantă pe carosabil după ploaie (fără ploaie vizibilă în cadru)
#   C6  — fir electric atârnând de la stâlp (pericol) — doar dacă e clar gol
#   C7  — semne de alunecare anvelope (marcaje lungi) fără vehicul în cadru
#   C8  — măști medicale / mănuși chirurgicale aruncate pe jos
#   C9  — recipient plastic tip bidon chimic industrial lăsat pe sol
#   C10 — oglindă / geam spart pe trotuar (cioburi)
#   D1  — manechin / statuie realistă confundabilă cu om (evită fețe reale)
#   D2  — umbrelă deschisă fixată pe sol (vânt puternic — static)
#   D3  — covor / preș mare rulat pe trotuar
#   D4  — cărucior copil răsturnat fără copil în cadru (sensibil — folosește rar)
#   D5  — trofee / obiect voluminos neobișnuit pe jos (ex. echipament sport mare)
#
# (Definițiile folosite efectiv în prompt sunt în dict ANOMALY_CATALOG — păstrează
#  comentariul și dict-ul aliniate când adaugi ID-uri noi.)
# =============================================================================

# Setare opțională în fișier: dacă nu e None și nu treci --anomaly_ids la CLI,
# se folosește această listă (tot format virgule).
DEFAULT_ANOMALY_IDS_FROM_FILE: str | None = None  # ex: "A1,A2,B5"

ANOMALY_CATALOG: dict[str, str] = {
    "A1": "Adult or teenager collapsed / lying on the ground (not standing), realistic clothing, CCTV quality.",
    "A2": "Large abandoned backpack or duffel bag on pavement or sidewalk, unattended, realistic fabric and straps.",
    "A3": "Bicycle or e-scooter lying flat on the ground (fallen), wheels visible, plausible walkway context.",
    "A4": "Traffic cone knocked over or small plastic barrier on asphalt/sidewalk.",
    "A5": "Cluster of litter / cardboard box / debris on ground near curb or walkway edge.",
    "A6": "Small ground fire or smoldering patch with subtle smoke (ground-level only, not a wall of flame).",
    "A7": "Rolling suitcase left alone on walkway, upright or slightly tilted.",
    "A8": "Skateboard left flat on pavement.",
    "A9": "Dark fluid spill / wet patch on concrete (oil-like sheen), no floating liquid.",
    "A10": "Fallen tree branch or long thin object on path.",
    "A11": "Closed umbrella abandoned on the ground.",
    "A12": "Cyclist helmet lying on the ground.",
    "A13": "Pair of shoes placed near road edge as if abandoned.",
    "A14": "Tied black garbage bag left on sidewalk.",
    "A15": "Small wooden pallet or crate on road or sidewalk.",
    "B1": "Person sitting on ground with back to wall (no chair), plausible street context.",
    "B2": "Shopping cart tipped over or left blocking path alone.",
    "B3": "Medium-sized dog without visible owner on sidewalk (surveillance anomaly).",
    "B4": "Motorcycle or scooter fallen on its side.",
    "B5": "Red fire extinguisher lying on ground near a building.",
    "B6": "Foldable metal ladder lying on sidewalk (out of place).",
    "B7": "Thick cable or fire hose snaking across walkway.",
    "B8": "Portable grill or metal drum on grass or concrete.",
    "B9": "Unusually large pile of dry leaves (maintenance anomaly).",
    "B10": "Temporary plastic road sign knocked over, reflective surface.",
    "B11": "Blanket or plastic tarp spread on ground like improvised camp.",
    "B12": "Car spare tire on sidewalk without vehicle in frame.",
    "B13": "Open car trunk lid or hatchback raised with no car body visible (edge case).",
    "B14": "Small unbarricaded roadworks hole or shallow trench.",
    "B15": "Stack of paving stones / concrete blocks on sidewalk.",
    "B16": "Person actively riding a bicycle in motion (pedaling, wheels turning), realistic street clothing, CCTV surveillance angle, plausible motion blur optional.",
    "C1": "Smoke wisps rising from a storm drain or manhole cover (subtle).",
    "C2": "Unusually strong glare patch on pavement (only if clearly an empty region).",
    "C3": "Fresh paint or whitewash spill patch on asphalt.",
    "C4": "Large pothole or sudden pavement break.",
    "C5": "Stagnant puddle on roadway with no rain visible in frame.",
    "C6": "Loose wire hanging from utility pole toward ground (only if background is empty).",
    "C7": "Long tire skid marks without vehicle present in frame.",
    "C8": "Discarded medical masks / gloves scattered on ground.",
    "C9": "Industrial-style plastic chemical jug left on ground.",
    "C10": "Broken mirror or glass shards on sidewalk.",
    "D1": "Realistic mannequin or statue on ground (avoid real human faces).",
    "D2": "Beach umbrella anchored open on pavement (unusual static object).",
    "D3": "Large rolled rug or mat abandoned on walkway.",
    "D4": "Baby stroller tipped over with no child visible in frame (use rarely, sensitive).",
    "D5": "Bulky sports equipment (e.g. goal net bag) left on path.",
}

DEFAULT_NEGATIVE_PROMPT = (
    "extra people, duplicate pedestrians, cartoon, floating objects, glowing edges, "
    "text, watermark, blurry, deformed limbs, low resolution"
)

# Anomalii care inpaintează o persoană (sau ciclist) — nu folosi „extra people” în negative.
ANOMALY_IDS_WITH_PERSON: frozenset[str] = frozenset({"A1", "B1", "B16", "D1"})

# Ciclist / corp întreg: bbox mai mare decât obiecte mici pe sol.
ANOMALY_IDS_LARGE_BBOX: frozenset[str] = frozenset({"A1", "B1", "B16", "B4", "D1", "D4"})

DEFAULT_NEGATIVE_PROMPT_WITH_PERSON = (
    "duplicate pedestrians, duplicate cyclists, second person, second bicycle, cartoon, "
    "floating objects, glowing edges, text, watermark, blurry, deformed limbs, low resolution"
)


def default_negative_prompt_for(active_ids: list[str]) -> str:
    if set(active_ids) & ANOMALY_IDS_WITH_PERSON:
        return DEFAULT_NEGATIVE_PROMPT_WITH_PERSON
    return DEFAULT_NEGATIVE_PROMPT


def build_system_instruction(active_ids: list[str]) -> str:
    """Construiește system prompt doar cu anomalii din `active_ids` (păstrate în ordinea dată)."""
    bullets = []
    for aid in active_ids:
        desc = ANOMALY_CATALOG[aid]
        bullets.append(f"   - **{aid}** — {desc}")
    catalog_block = "\n".join(bullets)

    ids_set = set(active_ids)
    has_person = bool(ids_set & ANOMALY_IDS_WITH_PERSON)
    has_cyclist = "B16" in ids_set
    needs_large_bbox = bool(ids_set & ANOMALY_IDS_LARGE_BBOX)

    if len(active_ids) == 1:
        pick_rule = (
            f"1. **You MUST use anomaly ID `{active_ids[0]}`** for this frame (no other types)."
        )
    else:
        pick_rule = (
            "1. **Choose exactly ONE anomaly type at random** for this frame, with "
            "**uniform probability** among ONLY these IDs (do not invent other types):"
        )

    if has_cyclist and len(active_ids) == 1:
        placement_extra = (
            "   - For **B16 (cyclist)**: the box must contain the **full rider + bicycle** "
            "(handlebars, frame, both wheels, pedaling legs). Place on walkway/asphalt with "
            "enough clearance; do not clip the cyclist at box edges."
        )
    elif has_person:
        placement_extra = (
            "   - For **person anomalies**: the box must contain the **full body** of exactly "
            "one person (or one cyclist+bike for B16), not only feet or shadow."
        )
    else:
        placement_extra = ""

    if has_person:
        sd_rule = (
            "3. **sd_prompt**: Highly detailed inpainting prompt for **exactly ONE** instance of "
            "the chosen anomaly (person/cyclist) plus local ground: materials, lighting, CCTV "
            "angle, mild compression noise, **photorealistic**. The prompt **must name** the "
            "subject (e.g. cyclist, bicycle, pedaling, person). **Do NOT** write \"no people\" or "
            "\"empty scene\". Avoid duplicates (**one** rider only). **Not floating**."
        )
        neg_rule = (
            "4. **negative_prompt**: Avoid **duplicates** (second cyclist, extra pedestrians), "
            "cartoon, text, watermark, heavy blur — but **do not** ban all humans."
        )
    else:
        sd_rule = (
            "3. **sd_prompt**: Highly detailed inpainting prompt for ONLY the chosen anomaly and "
            "the local ground/surface: materials (asphalt/concrete/tile), lighting, CCTV angle, "
            "mild compression noise, **photorealistic**, explicitly **no extra people** and "
            "**not floating**."
        )
        neg_rule = (
            "4. **negative_prompt**: Concise comma-separated list of what the inpainter must "
            "avoid (e.g. extra pedestrians, cartoon, text, watermark, blur)."
        )

    if needs_large_bbox:
        bbox_rule = (
            "5. **bounding_box**: JSON array `[x_min, y_min, x_max, y_max]` normalized 0.0–1.0. "
            "Target roughly **12%–45%** of image width and **15%–50%** of height so a full "
            "person, cyclist, or large object fits (adjust for perspective)."
        )
    else:
        bbox_rule = (
            "5. **bounding_box**: JSON array `[x_min, y_min, x_max, y_max]` normalized 0.0–1.0. "
            "Target roughly **6%–30%** of image width and **6%–30%** of height (adjust for perspective)."
        )

    return f"""
You are an expert CCTV / surveillance scene analyst. You receive ONE static frame (street, campus, walkway, etc.).

## Your task
{pick_rule}
{catalog_block}

2. **Placement rule (critical):** The anomaly must sit in a **visually empty** region: **clear ground, floor, asphalt, grass strip, or wide walkway** where the inpainted content will not cover an existing **person, face, vehicle, bicycle in use, dog, stroller, text/sign, doorway, window with people, or dense foliage** already in the frame.
   - First mentally scan the image: identify where pedestrians and major objects are.
   - Place the bounding box **only** in free space between those regions, or in the **least cluttered** plausible ground patch if the scene is busy.
   - The box should tightly frame where the anomaly will appear (not the whole image).
{placement_extra}

{sd_rule}

{neg_rule}

{bbox_rule}

## Output format
Respond with **ONLY** valid JSON (no markdown fences, no commentary), exactly these keys:
- `"sd_prompt"`: string.
- `"negative_prompt"`: string.
- `"bounding_box"`: array of 4 floats `[x_min, y_min, x_max, y_max]`.
"""


def parse_anomaly_ids_arg(cli_value: str | None, file_default: str | None) -> list[str]:
    """
    Întoarce lista de ID-uri (chei ANOMALY_CATALOG). Ordinea păstrată = ordinea din string.
    """
    raw = (cli_value if cli_value is not None and str(cli_value).strip() else None) or file_default
    if not raw or not str(raw).strip():
        return list(ANOMALY_CATALOG.keys())

    seen: set[str] = set()
    ordered: list[str] = []
    for part in str(raw).split(","):
        k = part.strip().upper()
        if not k:
            continue
        if k not in ANOMALY_CATALOG:
            print(f"⚠️  ID necunoscut (ignorat): {k}")
            continue
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    if not ordered:
        print("⚠️  Nicio intrare validă în --anomaly_ids; folosesc tot catalogul.")
        return list(ANOMALY_CATALOG.keys())
    return ordered


DATASET_DEFAULTS = {
    "avenue": {"data_root": "./Avenue_Dataset", "extensions": (".png", ".jpg", ".jpeg")},
    "ucsd": {"data_root": "./UCSD_Dataset", "extensions": (".tif", ".tiff")},
}


def _image_part_for_gemini(image_path: str) -> dict:
    """
    Gemini API respinge image/tiff (400 Unsupported MIME type).
    Convertim mereu la PNG în memorie (merge și pentru png/jpg).
    """
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"mime_type": "image/png", "data": buf.getvalue()}


def get_anomaly_context_from_vlm(
    image_path: str,
    system_instruction: str,
    active_ids: list[str] | None = None,
):
    model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_instruction)
    image_part = _image_part_for_gemini(image_path)

    if active_ids and len(active_ids) == 1:
        user_text = (
            f"Analyze the frame. You MUST use anomaly ID {active_ids[0]} only. "
            "Choose an empty ground region that does not overlap existing people or major "
            "objects, then output the JSON only."
        )
    else:
        user_text = (
            "Analyze the frame. Pick ONE anomaly type at random from the ALLOWED IDs "
            "in your instructions only, choose an empty ground region that does not overlap "
            "people or major existing objects, then output the JSON only."
        )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                [
                    image_part,
                    user_text,
                ],
                generation_config=genai.GenerationConfig(response_mime_type="application/json"),
            )
            data = json.loads(response.text)
            return data["sd_prompt"], data.get("negative_prompt", ""), data["bounding_box"]

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Quota exceeded" in error_msg:
                print(
                    f"Limită rată API. Pauză 60s... (încercarea {attempt + 1}/{max_retries})"
                )
                time.sleep(60)
            else:
                print(f"Eroare VLM: {error_msg}")
                return None, None, None

    print("❌ Prea multe încercări VLM pentru acest cadru.")
    return None, None, None


def create_mask_image(width: int, height: int, bbox):
    mask = np.zeros((height, width), dtype=np.uint8)

    x1 = int(bbox[0] * width)
    y1 = int(bbox[1] * height)
    x2 = int(bbox[2] * width)
    y2 = int(bbox[3] * height)

    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    blur_radius = int(width * 0.08)
    if blur_radius % 2 == 0:
        blur_radius += 1

    mask_blurred = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
    return Image.fromarray(mask_blurred)


def variant_filename(frame_name: str, variant_idx: int) -> str:
    """Ex.: 0042.png + v3 → 0042_v03.png"""
    stem, ext = os.path.splitext(frame_name)
    return f"{stem}_v{variant_idx:02d}{ext}"


def _inpaint_generator(seed: int | None, run_i: int, variant_i: int) -> torch.Generator | None:
    if seed is None:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + run_i * 10_000 + variant_i * 997)
    return gen


def setup_inpaint_pipe(pipe) -> str:
    """
    Plasează pipeline-ul pe dispozitiv.
    `enable_model_cpu_offload` necesită pachetul `accelerate`; altfel folosește CUDA direct.
    """
    if torch.cuda.is_available():
        try:
            import accelerate  # noqa: F401

            pipe.enable_model_cpu_offload()
            return "cpu_offload (accelerate)"
        except (ImportError, RuntimeError) as e:
            print(f"  ⚠️ cpu_offload indisponibil ({e}); folosesc CUDA direct.")

        pipe.to("cuda", torch_dtype=torch.float16)
        for fn_name in ("enable_attention_slicing", "enable_vae_slicing"):
            fn = getattr(pipe, fn_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        return "cuda"

    pipe.to("cpu")
    return "cpu"


def run_inpaint_variants(
    pipe,
    init_image: Image.Image,
    mask_image: Image.Image,
    sd_prompt: str,
    neg_prompt: str,
    num_variants: int,
    seed: int | None,
    run_i: int,
) -> list[Image.Image]:
    """Rulează Juggernaut de `num_variants` ori (seed diferit → variații vizuale)."""
    results: list[Image.Image] = []
    for variant_i in range(1, num_variants + 1):
        generator = _inpaint_generator(seed, run_i, variant_i)
        kwargs: dict = {
            "prompt": sd_prompt,
            "negative_prompt": neg_prompt,
            "image": init_image,
            "mask_image": mask_image,
            "num_inference_steps": 30,
            "guidance_scale": 7.0,
        }
        if generator is not None:
            kwargs["generator"] = generator
        results.append(pipe(**kwargs).images[0])
    return results


def discover_train_frames(frames_dir: str, extensions: tuple[str, ...]) -> list[tuple[str, str]]:
    """Liste (subfolder, filename) pentru fișiere din train/frames/*/."""
    pairs: list[tuple[str, str]] = []
    if not os.path.isdir(frames_dir):
        return pairs
    for sub in sorted(os.listdir(frames_dir)):
        folder = os.path.join(frames_dir, sub)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            low = name.lower()
            if any(low.endswith(ext) for ext in extensions):
                pairs.append((sub, name))
    return pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gemini + inpainting → frames_abnormal & masks_abnormal (Avenue sau UCSD)."
    )
    p.add_argument(
        "--dataset",
        type=str,
        choices=("avenue", "ucsd"),
        required=True,
        help="Tip layout: avenue (png/jpg) sau ucsd (tif).",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Rădăcina dataset-ului. Dacă lipsește: ./Avenue_Dataset sau ./UCSD_Dataset după --dataset.",
    )
    p.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Câte apeluri Gemini (cadre aleatoare). Fiecare run → --inpaint_variants ieșiri Juggernaut.",
    )
    p.add_argument(
        "--inpaint_variants",
        type=int,
        default=5,
        help="Câte generări Juggernaut per apel Gemini (seed diferit). Ex.: num_runs=5, inpaint_variants=5 → 25 anomalii.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--mirror_normals",
        action="store_true",
        default=True,
        help="Copiază cadrul normal în train/frames cu sufix _vNN ca să se potrivească la antrenare (implicit activ).",
    )
    p.add_argument(
        "--no_mirror_normals",
        action="store_false",
        dest="mirror_normals",
        help="Nu copia cadre normale sufixate în train/frames (doar frames_abnormal / masks_abnormal).",
    )
    p.add_argument(
        "--model_id",
        type=str,
        default="RunDiffusion/Juggernaut-XL-v9",
    )
    p.add_argument(
        "--frames_abnormal_dir",
        type=str,
        default=None,
        help="Opțional: folderul rădăcină `frames_abnormal` (conține subfoldere ca Train001). "
        "Implicit: <data_root>/train/frames_abnormal",
    )
    p.add_argument(
        "--masks_dir",
        type=str,
        default=None,
        help="Opțional: folderul rădăcină `masks_abnormal` (aceeași structură de subfoldere). "
        "Implicit: <data_root>/train/masks_abnormal. Ex. Colab: "
        '"/content/drive/MyDrive/Licenta/export/train/masks_abnormal"',
    )
    p.add_argument(
        "--anomaly_ids",
        type=str,
        default=None,
        help="ID-uri din ANOMALY_CATALOG, separate prin virgulă (ex. A1,A3,B2). "
        "Implicit: toate. Alternativ: setează DEFAULT_ANOMALY_IDS_FROM_FILE în fișier.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]
    data_root = os.path.abspath(
        os.path.expanduser(args.data_root or defaults["data_root"])
    )
    extensions = defaults["extensions"]
    rng = random.Random(args.seed)

    train_frames_dir = os.path.join(data_root, "train", "frames")
    abnormal_frames_dir = os.path.abspath(
        os.path.expanduser(
            args.frames_abnormal_dir
            or os.path.join(data_root, "train", "frames_abnormal")
        )
    )
    abnormal_masks_dir = os.path.abspath(
        os.path.expanduser(
            args.masks_dir or os.path.join(data_root, "train", "masks_abnormal")
        )
    )

    os.makedirs(abnormal_frames_dir, exist_ok=True)
    os.makedirs(abnormal_masks_dir, exist_ok=True)

    all_pairs = discover_train_frames(train_frames_dir, extensions)
    if not all_pairs:
        print(f"❌ Nu am găsit cadre în {train_frames_dir} cu extensiile {extensions}")
        sys.exit(1)

    total_planned = args.num_runs * args.inpaint_variants
    print(f"Dataset: {args.dataset} | root: {data_root}")
    print(
        f"Cadre indexate: {len(all_pairs)} | "
        f"{args.num_runs} apeluri Gemini × {args.inpaint_variants} inpaint = {total_planned} anomalii"
    )
    print(f"Ieșire cadre anormale: {abnormal_frames_dir}")
    print(f"Ieșire măști:        {abnormal_masks_dir}")
    if args.mirror_normals:
        print(f"Mirror normale:      da (copie în {train_frames_dir} cu sufix _vNN)")

    active_anomaly_ids = parse_anomaly_ids_arg(args.anomaly_ids, DEFAULT_ANOMALY_IDS_FROM_FILE)
    system_instruction = build_system_instruction(active_anomaly_ids)
    print(f"ID-uri anomalii active ({len(active_anomaly_ids)}): {', '.join(active_anomaly_ids)}")

    print("Încărcăm modelul de inpainting...")
    pipe = AutoPipelineForInpainting.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    device_mode = setup_inpaint_pipe(pipe)
    print(f"Pipeline: {device_mode}")

    ok = 0
    for run_i in range(args.num_runs):
        sub, frame_name = rng.choice(all_pairs)
        original_frame_path = os.path.join(train_frames_dir, sub, frame_name)
        print(
            f"\n[run {run_i + 1}/{args.num_runs}] -> {sub}/{frame_name} "
            f"({args.inpaint_variants} variante Juggernaut)"
        )

        sd_prompt, neg_prompt, bbox = get_anomaly_context_from_vlm(
            original_frame_path, system_instruction, active_anomaly_ids
        )
        if not sd_prompt or not bbox:
            print("  Sărit: VLM fără rezultat valid.")
            continue
        neg_prompt = (neg_prompt or "").strip() or default_negative_prompt_for(active_anomaly_ids)
        print(f"  VLM bbox: {bbox}")
        print(f"  VLM sd_prompt: {sd_prompt[:220]}{'...' if len(sd_prompt) > 220 else ''}")

        init_image = Image.open(original_frame_path).convert("RGB")
        width, height = init_image.size
        mask_image = create_mask_image(width, height, bbox)

        variant_images = run_inpaint_variants(
            pipe,
            init_image,
            mask_image,
            sd_prompt,
            neg_prompt,
            args.inpaint_variants,
            args.seed,
            run_i,
        )

        os.makedirs(os.path.join(abnormal_frames_dir, sub), exist_ok=True)
        os.makedirs(os.path.join(abnormal_masks_dir, sub), exist_ok=True)
        if args.mirror_normals:
            os.makedirs(os.path.join(train_frames_dir, sub), exist_ok=True)

        for variant_i, result_image in enumerate(variant_images, start=1):
            out_name = variant_filename(frame_name, variant_i)
            dest_frame_path = os.path.join(abnormal_frames_dir, sub, out_name)
            dest_mask_path = os.path.join(abnormal_masks_dir, sub, out_name)

            result_image.save(dest_frame_path)
            mask_image.save(dest_mask_path)

            if args.mirror_normals:
                mirrored_normal = os.path.join(train_frames_dir, sub, out_name)
                if not os.path.isfile(mirrored_normal):
                    shutil.copy2(original_frame_path, mirrored_normal)

            print(f"  ✅ v{variant_i:02d} frame: {dest_frame_path}")
            print(f"       mask: {dest_mask_path}")
            ok += 1

    print(f"\nGata: {ok}/{total_planned} anomalii reușite ({args.num_runs} apeluri Gemini).")
    print(f"  frames_abnormal → {abnormal_frames_dir}")
    print(f"  masks_abnormal  → {abnormal_masks_dir}")


if __name__ == "__main__":
    main()
