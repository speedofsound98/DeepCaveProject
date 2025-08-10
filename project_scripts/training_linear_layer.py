import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from transformers import AutoImageProcessor, Dinov2Model
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

"""training_linear_layer_with_error_handler.py
Trains a linear classification head on top of frozen DINOv2 image embeddings, 
with robust error handling.

- Loads image dataset, applies preprocessing, extracts DINOv2 embeddings 
(frozen backbone).
- Optimizes a linear layer (e.g., nn.Linear) using cross-entropy loss.
- Skips/logs corrupted or unreadable images to avoid aborting the run.
- Saves: trained head weights, label index mapping, and bad-images logs.

CLI example:
    python training_linear_layer_with_error_handler.py --data <dataset_root> --
    epochs 10 --batch-size 64

Outputs:
- model_head.pt / state_dict: trained linear head weights
- label_to_index.json / index_to_label.json: class mappings
- bad_images.txt / bad_images.json: files that failed to load or process
"""

# >>> ADDED
import logging, traceback, time
from PIL import ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # let PIL read truncated JPEGs
# instead of crashing

BAD_IMG_LOG_JSON = "bad_images.json"
BAD_IMG_LOG_TXT  = "bad_images.txt"
GLOBAL_BAD_LIST  = []  # will be filled as we go
# >>>

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ─── CONFIGURATION ────────────────────────────────────────────────────────────
AIRTABLE_BASE_ID = "appFHLlsLJS6Brzer"
AIRTABLE_TABLE = "tbleSBboDC5DMiJLw"
AIRTABLE_TOKEN = "patCTj2g6nxRzwdm7.250196264f1fbcd0384c1d8f4bb5298fafd2e25fe04f263c4e5d906df8509633"

# Root folder where every Airtable “path” is relative to
ROOT_PATH = r"G:\.shortcut-targets-by-id\1oVtyQFSpu1xR4VAK7YGbOo7cc2uyF1lm\בסיס נתונים מערות\גיבוי תיקים"

LOCAL_MODEL_DIR = "./local_dinov2"
OUTPUT_HEAD_PATH = "cave_classifier.pth"
OUTPUT_MAP_PATH = "cave_to_idx.json"

# Paths for caching embeddings
EMB_CACHE_PATH = "all_embeddings.pt"
EMB_MAP_PATH = "emb_path_label_map.json"

BATCH_SIZE = 64
NUM_EPOCHS = 118
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ─── DINOv2 EMBEDDING HELPERS ─────────────────────────────────────────────────
def get_dino_model(local_dir):
    """
    Load DINOv2 in eval mode and return (model, transform).
    """
    processor = AutoImageProcessor.from_pretrained(local_dir, use_fast=True)
    dino_model = Dinov2Model.from_pretrained(local_dir).to(DEVICE)
    dino_model.eval()

    transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Resize(244),
        T.CenterCrop(224),
        T.Normalize([0.5], [0.5]),
    ])

    return dino_model, transform


def load_and_embed(path: str, dino_model, transform) -> torch.Tensor:
    """
    Given an image path, load it, apply transforms,
    and return its 768-D embedding.
    """
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
    with torch.no_grad():
        feats = dino_model(t).last_hidden_state   # [1,197,768]
        emb = feats.mean(dim=1)                 # [1,768]
    return emb.squeeze(0)                        # [768]


# >>> ADDED
def safe_load_and_embed(path: str, dino_model, transform):
    """
    Wraps load_and_embed with try/except so a single bad image won't kill the run.
    Returns Tensor or None on failure.
    """
    try:
        return load_and_embed(path, dino_model, transform)
    except (OSError, UnidentifiedImageError, ValueError, RuntimeError) as e:
        GLOBAL_BAD_LIST.append({"path": path, "error": repr(e)})
        return None
    except Exception as e:
        # Catch-all, but still log full traceback
        GLOBAL_BAD_LIST.append({"path": path, "error": repr(e), "traceback": traceback.format_exc()})
        return None
# >>>


# ─── CACHING EMBEDDINGS & MAPPING ─────────────────────────────────────────────
def build_and_cache_embeddings():
    all_recs = get_all_records(AIRTABLE_BASE_ID, AIRTABLE_TABLE, AIRTABLE_TOKEN)

    filtered_pairs = []
    cave_names = set()
    for rec in all_recs:
        f = rec.get("fields", {})
        if f.get("type") != "פתח":
            continue
        if f.get("test_photo") is True:
            continue
        if f.get("bad_photo") is True:
            continue
        cave_name = f.get("cave_name")
        rel_path  = f.get("path")
        if not cave_name or not rel_path:
            # >>> ADDED
            GLOBAL_BAD_LIST.append({"path": rel_path, "error": "path does not exist"})
            # >>>
            continue

        rel_norm = os.path.normpath(rel_path)
        full_path = os.path.join(ROOT_PATH, rel_norm)
        if not os.path.isfile(full_path):
            continue

        cave_names.add(cave_name)
        filtered_pairs.append((full_path, cave_name))

    if not filtered_pairs:
        print("No valid entrance‐image records found. Exiting.")
        exit()

    # Build cave_to_idx mapping
    sorted_caves = sorted(cave_names)
    cave_to_idx = {c: i for i, c in enumerate(sorted_caves)}
    idx_to_cave = {i: c for c, i in cave_to_idx.items()}
    num_classes = len(cave_to_idx)

    # Save mapping
    with open(OUTPUT_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(cave_to_idx, f, ensure_ascii=False, indent=2)
    print(f"Found {num_classes} distinct caves. Saved mapping to {OUTPUT_MAP_PATH}")

    # If already cached, skip
    if os.path.isfile(EMB_CACHE_PATH):
        print(f"\nEmbeddings cache already exists at {EMB_CACHE_PATH}")
        return filtered_pairs, cave_to_idx

    print("\nCaching all embeddings…")
    dino_model, transform = get_dino_model(LOCAL_MODEL_DIR)

    embeddings_list = []
    path_label_map  = []  # will store [ [img_path, label], ... ]

    # >>> ADDED
    bad_count = 0

    for full_path, cave_name in tqdm(filtered_pairs, desc="Caching Embeddings"):
        label = cave_to_idx[cave_name]
        # emb = load_and_embed(full_path, dino_model, transform)  # [768]
        emb = safe_load_and_embed(full_path, dino_model, transform)
        if emb is None:
            bad_count += 1
            continue
        embeddings_list.append(emb.cpu())
        path_label_map.append([full_path, label])
    # >>> ADDED
    if GLOBAL_BAD_LIST:
        with open(BAD_IMG_LOG_JSON, "w", encoding="utf-8") as f:
            json.dump(GLOBAL_BAD_LIST, f, ensure_ascii=False, indent=2)
        with open(BAD_IMG_LOG_TXT, "w", encoding="utf-8") as f:
            for item in GLOBAL_BAD_LIST:
                f.write(f"{item['path']} :: {item['error']}\n")
        print(f"⚠️ Skipped {len(GLOBAL_BAD_LIST)} problematic images. Logged to {BAD_IMG_LOG_JSON} / {BAD_IMG_LOG_TXT}")

    # Stack into one [N,768] tensor
    all_embeddings = torch.stack(embeddings_list, dim=0)  # CPU tensor

    # Save tensor and mapping
    torch.save(all_embeddings, EMB_CACHE_PATH)
    with open(EMB_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(path_label_map, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(path_label_map)} embeddings to {EMB_CACHE_PATH}")
    return filtered_pairs, cave_to_idx


# ─── DATASET & DATALOADER ─────────────────────────────────────────────────────
class CachedEmbeddingDataset(Dataset):
    def __init__(self, emb_cache_path: str, map_json: str):
        """
        emb_cache_path: path to all_embeddings.pt (shape [N,768]).
        map_json: path to JSON list [ [img_path, label], ... ] in same order.
        """
        self.embeddings = torch.load(emb_cache_path)  # [N,768] on CPU
        with open(map_json, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        self.labels = [p[1] for p in pairs]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        label = self.labels[idx]
        return emb, label


# ─── MODEL: Linear head definition ────────────────────────────────────────────
class LinearHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(x)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Build & cache embeddings (if not already done)
    filtered_pairs, cave_to_idx = build_and_cache_embeddings()
    num_classes = len(cave_to_idx)

    # 2) Build dataset & dataloader
    train_dataset = CachedEmbeddingDataset(EMB_CACHE_PATH, EMB_MAP_PATH)
    num_samples = len(train_dataset)
    print(f"\n🛠️  Training on {num_samples} cached embeddings for {num_classes} classes.\n")

    # On Windows, either set num_workers=0 or wrap in main (we did both here).
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,      # ← set to 0 to avoid multiprocessing issues
        pin_memory=False    # no GPU accelerator found warning
    )

    # 3) Build model, loss, optimizer
    head = LinearHead(embed_dim=768, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(head.parameters(), lr=LR)

    # 4) Training loop with checkpoints
    for epoch in range(1, NUM_EPOCHS + 1):
        head.train()
        total_loss = 0.0

        for emb_batch, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            emb_batch = emb_batch.to(DEVICE, non_blocking=True)  # [B,768]
            labels    = torch.tensor(labels, dtype=torch.long, device=DEVICE)

            optimizer.zero_grad()
            logits = head(emb_batch)                   # [B, num_classes]
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * emb_batch.size(0)

        avg_loss = total_loss / num_samples
        print(f"→ Epoch {epoch} average loss: {avg_loss:.4f}")

        # Save checkpoint after each epoch
        ckpt = {
            "epoch": epoch,
            "model_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "avg_loss": avg_loss
        }
        torch.save(ckpt, f"checkpoint_epoch{epoch}.pth")
        print(f"→ Saved checkpoint_epoch{epoch}.pth\n")

    # 5) Save final model
    torch.save(head.state_dict(), OUTPUT_HEAD_PATH)
    print(f"\n✅ Trained linear head saved to {OUTPUT_HEAD_PATH}")


# ─── TRAINING LOOP FOR LINEAR HEAD ────────────────────────────────────────────
head = LinearHead(embed_dim=768, num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(head.parameters(), lr=LR)

for epoch in range(1, NUM_EPOCHS + 1):
    head.train()
    total_loss = 0.0

    for emb_batch, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        # 1) Move data to GPU/CPU
        emb_batch = emb_batch.to(DEVICE)   # shape: [batch_size, 768]
        labels    = labels.to(DEVICE)      # shape: [batch_size]

        # 2) Forward pass: compute logits and loss
        logits = head(emb_batch)           # shape: [batch_size, num_classes]
        loss   = criterion(logits, labels)

        # 3) Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * emb_batch.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"→ Epoch {epoch} average loss: {avg_loss:.4f}")
    # 4) Save checkpoint at each epoch
    torch.save(head.state_dict(), f"checkpoint_epoch{epoch}.pth")
