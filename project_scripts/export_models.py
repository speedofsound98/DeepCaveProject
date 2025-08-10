import torch, json, sys, traceback
from torch.utils.mobile_optimizer import optimize_for_mobile
from transformers import Dinov2Model

"""export_models.py
Exports the DINOv2 + linear head model for deployment (e.g., TorchScript).

- Loads the backbone and trained head.
- Traces or scripts the model to a serialized format (TorchScript) suitable for 
offline/mobile use.
- Optionally performs simple sanity checks on a dummy input.

CLI example:
    python export_models.py --head-path model_head.pt --out model.ts
"""


LOCAL_MODEL_DIR = "local_dinov2"
HEAD_WEIGHTS = "cave_classifier.pth"
CAVE_MAP_JSON = "cave_to_idx.json"


class LinearHead(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.cls = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.cls(x)


class DinoWrapper(torch.nn.Module):
    def __init__(self, dino):
        super().__init__()
        self.dino = dino

    def forward(self, x):
        with torch.no_grad():
            feats = self.dino(x).last_hidden_state.detach()   # <— detach!
            return feats.mean(dim=1)


def main():
    try:
        torch.set_grad_enabled(False)  # global off

        print("Loading DINO from", LOCAL_MODEL_DIR)
        dino = Dinov2Model.from_pretrained(LOCAL_MODEL_DIR)
        dino.eval()
        embed_dim = dino.config.hidden_size
        print("hidden_size:", embed_dim)

        wrapper = DinoWrapper(dino)
        example = torch.randn(1,3,224,224)

        print("Tracing DINO…")
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced = optimize_for_mobile(traced)
        traced._save_for_lite_interpreter("dino.ptl")
        print("Saved dino.ptl")

        with open(CAVE_MAP_JSON,"r",encoding="utf-8") as f:
            cave_map = json.load(f)
        num_classes = len(cave_map)
        print("num_classes:", num_classes)

        head = LinearHead(embed_dim, num_classes)
        sd = torch.load(HEAD_WEIGHTS, map_location="cpu")
        missing, unexpected = head.load_state_dict(sd, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        head.eval()

        print("Tracing head…")
        head_traced = torch.jit.trace(head, torch.randn(1, embed_dim), strict=False)
        head_traced = optimize_for_mobile(head_traced)
        head_traced._save_for_lite_interpreter("head.ptl")
        print("Saved head.ptl")

        print("DONE ✅")
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
