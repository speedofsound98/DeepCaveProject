import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from transformers import AutoImageProcessor, Dinov2Model
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

"""final_inference.py
Runs the DeepCave inference pipeline.

- Loads DINOv2 as a feature extractor and a trained linear classification head.
- Preprocesses images with torchvision / PIL transforms.
- Performs forward pass to produce class probabilities and top-k predictions.
- (If configured) fetches cave metadata via HTTP and displays results.
- May include simple Tkinter UI elements for visual feedback.

Usage:
    python final_inference.py --input <path-to-image-or-folder> [--topk 5] 
    [--model <path>]

Notes:
- Requires the same preprocessing used during training.
- Ensure the weights for the linear head match the DINOv2 variant used.
"""

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ─── CONFIGURATION ────────────────────────────────────────────────────────────
AIRTABLE_BASE_ID = "appFHLlsLJS6Brzer"
AIRTABLE_TABLE = "tbleSBboDC5DMiJLw"
AIRTABLE_TOKEN = "patCTj2g6nxRzwdm7.250196264f1fbcd0384c1d8f4bb5298fafd2e25fe04f263c4e5d906df8509633"

# Root folder where every Airtable “path” is relative to
ROOT_PATH = r"G:\.shortcut-targets-by-id\1oVtyQFSpu1xR4VAK7YGbOo7cc2uyF1lm\בסיס נתונים מערות\גיבוי תיקים"

LOCAL_MODEL_DIR = "./local_dinov2"
LINEAR_HEAD_PATH = "checkpoint_epoch118.pth"
CAVE_TO_IDX_PATH = "cave_to_idx.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 5   # compute Top‐1 through Top‐5


# ─── AIRTABLE HELPER ──────────────────────────────────────────────────────────
def get_all_records(base_id, table_id, token):
    """
    Fetch all pages of records from the Airtable table.
    """
    url = f"https://api.airtable.com/v0/{base_id}/{table_id}"
    headers = {"Authorization": f"Bearer {token}"}
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(3, backoff_factor=0.3)))

    records, offset = [], None
    while True:
        params = {}
        if offset:
            params["offset"] = offset
        resp = session.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        records.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break

    return records


# ─── LOAD DINOv2 + TRANSFORMS ─────────────────────────────────────────────────
def load_dino_and_transform(local_dir):
    """
    Returns a frozen DINOv2 model and the image transform.
    """
    _ = AutoImageProcessor.from_pretrained(local_dir, use_fast=True)
    dino_model = Dinov2Model.from_pretrained(local_dir).to(DEVICE)
    dino_model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Resize(244),
        T.CenterCrop(224),
        T.Normalize([0.5], [0.5]),
    ])
    return dino_model, transform


def embed_image(path: str, dino_model, transform) -> torch.Tensor:
    """
    Load an image from `path`, apply transforms, pass through DINOv2,
    and return the 768-D embedding (CPU tensor).
    """
    img = Image.open(path).convert("RGB")
    t   = transform(img).unsqueeze(0).to(DEVICE)   # [1,3,224,224]
    with torch.no_grad():
        feats = dino_model(t).last_hidden_state     # [1,197,768]
        emb   = feats.mean(dim=1).squeeze(0)         # [768]
    return emb.cpu()


# ─── LINEAR HEAD DEFINITION ───────────────────────────────────────────────────
class LinearHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(x)


# ─── EVALUATION ON TEST IMAGES ────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Load cave→idx mapping, build idx→cave
    with open(CAVE_TO_IDX_PATH, "r", encoding="utf-8") as f:
        cave_to_idx = json.load(f)
    idx_to_cave = {v: k for k, v in cave_to_idx.items()}
    num_classes = len(cave_to_idx)

    # 2) Load linear head
    head = LinearHead(embed_dim=768, num_classes=num_classes).to(DEVICE)
    head.load_state_dict(torch.load(LINEAR_HEAD_PATH, map_location=DEVICE))
    head.eval()

    # 3) Load DINOv2 model + transform
    dino_model, transform = load_dino_and_transform(LOCAL_MODEL_DIR)

    # 4) Fetch Airtable records and filter test images
    all_recs  = get_all_records(AIRTABLE_BASE_ID, AIRTABLE_TABLE, AIRTABLE_TOKEN)
    test_recs = [r for r in all_recs if r.get("fields", {}).get("test_photo") is True]
    print(f"Found {len(test_recs)} test images.\n")

    results = []
    for rec in tqdm(test_recs, desc="Evaluating test set"):
        f = rec.get("fields", {})
        cave_true = f.get("cave_name")
        rel_path = f.get("path")  # e.g. "17-07/בור מים מרחב ערד21/IMG_8065.JPG"
        rel_norm = os.path.normpath(rel_path)
        full_path = os.path.join(ROOT_PATH, rel_norm)

        if not os.path.isfile(full_path):
            results.append({
                "image": rel_path,
                "true":  cave_true,
                "pred_topk": [],
                "ok_top1": False,
                "ok_top2": False,
                "ok_top3": False,
                "ok_top4": False,
                "ok_top5": False,
                "note":  "file not found"
            })
            continue

        # 5) Embed test image
        emb = embed_image(full_path, dino_model, transform).to(DEVICE)  # [768]

        # 6) Predict top-K via linear head
        with torch.no_grad():
            logits = head(emb.unsqueeze(0))               # [1, num_classes]
            probs  = F.softmax(logits, dim=-1).squeeze(0) # [num_classes]

        # 7) Get top-K indices and normalized confidences
        topk_vals, topk_idxs = torch.topk(probs, k=TOP_K)
        topk_vals = topk_vals.cpu().tolist()
        topk_idxs = topk_idxs.cpu().tolist()

        # Normalize top-K scores so they sum to 1.0
        s = sum(topk_vals) + 1e-12
        topk_norm = [v / s for v in topk_vals]

        # Map idx→cave_name
        topk_preds = [(idx_to_cave[idx], topk_norm[i]) for i, idx in enumerate(topk_idxs)]

        # 8) Check Top‐1 through Top‐5 correctness
        topk_names = [name for (name, _) in topk_preds]
        ok_top1 = (topk_names[0] == cave_true)
        ok_top2 = (cave_true in topk_names[:2])
        ok_top3 = (cave_true in topk_names[:3])
        ok_top4 = (cave_true in topk_names[:4])
        ok_top5 = (cave_true in topk_names[:5])

        results.append({
            "image": rel_path,
            "true":  cave_true,
            "pred_topk": topk_preds,  # list of (cave_name, normalized_prob)
            "ok_top1": ok_top1,
            "ok_top2": ok_top2,
            "ok_top3": ok_top3,
            "ok_top4": ok_top4,
            "ok_top5": ok_top5,
            "note":  ""
        })

    # 9) Print results to console
    header = (
        f"{'Image':<40} {'True':<20} {'Top-1':<20} {'Conf(Top-1)':<10} {'1?':<3}"
        f"{'Top-5 (name,%)':<60} {'5?':<3}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if r["note"] == "file not found":
            mark1 = "❌"
            mark5 = "❌"
            top1_name = "-"
            top1_conf = 0.0
            top5_str  = "N/A"
        else:
            top1_name, top1_conf = r["pred_topk"][0]
            mark1 = "✅" if r["ok_top1"] else "❌"
            mark5 = "✅" if r["ok_top5"] else "❌"
            top5_str = ", ".join(
                [f"{name} ({conf:.2%})" for name, conf in r["pred_topk"]]
            )

        print(
            f"{r['image']:<40} "
            f"{(r['true'] or '-'): <20} "
            f"{top1_name:<20} "
            f"{top1_conf:.2%}   "
            f"{mark1:<3} "
            f"{top5_str:<60} "
            f"{mark5}"
        )

    # 10) Compute overall Top-1 through Top-5 accuracy
    valid = [r for r in results if r["note"] == ""]
    total = len(valid)

    correct_1 = sum(r["ok_top1"] for r in valid)
    correct_2 = sum(r["ok_top2"] for r in valid)
    correct_3 = sum(r["ok_top3"] for r in valid)
    correct_4 = sum(r["ok_top4"] for r in valid)
    correct_5 = sum(r["ok_top5"] for r in valid)

    acc1 = (correct_1 / total * 100) if total else 0.0
    acc2 = (correct_2 / total * 100) if total else 0.0
    acc3 = (correct_3 / total * 100) if total else 0.0
    acc4 = (correct_4 / total * 100) if total else 0.0
    acc5 = (correct_5 / total * 100) if total else 0.0

    print(f"\nOverall Top‐1 Accuracy: {correct_1}/{total} = {acc1:.2f}%")
    print(f"Overall Top‐2 Accuracy: {correct_2}/{total} = {acc2:.2f}%")
    print(f"Overall Top‐3 Accuracy: {correct_3}/{total} = {acc3:.2f}%")
    print(f"Overall Top‐4 Accuracy: {correct_4}/{total} = {acc4:.2f}%")
    print(f"Overall Top‐5 Accuracy: {correct_5}/{total} = {acc5:.2f}%")
