"""
Geometric consistency metrics and quality gate for seasonal augmentation.

Metrics:
  - DINOv2 cosine similarity (CLS + patch-level)
  - Edge F1 (Canny with pixel tolerance)
  - SSIM on edge maps

Usage as a quality gate during generation:

    gate = QualityGate(device="cuda")
    passed, metrics = gate.check(original_pil, generated_pil)
    if not passed:
        # reject or retry
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ─── DINOv2 ──────────────────────────────────────────────────────────────────


class DINOv2Similarity:
    """Structural similarity using DINOv2 features (style-invariant)."""

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        self.device = device
        self.patch_size = 14
        self.target_size = 518  # 37 * 14

        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name, verbose=False,
        )
        self.model = self.model.to(device).eval()
        self.transform = self._build_transform()

    def _build_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(
                (self.target_size, self.target_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def compute(self, original: Image.Image, generated: Image.Image) -> dict:
        """Compute CLS and patch-level cosine similarity.

        Returns dict with:
            dino_cls_sim: float — global structural similarity [0, 1]
            dino_patch_sim_mean: float — mean patch similarity [0, 1]
            dino_patch_sim_min: float — worst patch similarity
            _patch_sim_map: np.ndarray — spatial heatmap (37x37)
        """
        imgs = torch.stack([
            self.transform(original),
            self.transform(generated),
        ]).to(self.device)

        features = self.model.forward_features(imgs)
        cls_tokens = features["x_norm_clstoken"]       # (2, D)
        patch_tokens = features["x_norm_patchtokens"]   # (2, N, D)

        cls_sim = F.cosine_similarity(
            cls_tokens[0:1], cls_tokens[1:2], dim=-1
        ).item()

        patch_sim = F.cosine_similarity(
            patch_tokens[0:1], patch_tokens[1:2], dim=-1
        ).squeeze(0).cpu().numpy()  # (N,)

        n = self.target_size // self.patch_size
        patch_map = patch_sim.reshape(n, n)

        return {
            "dino_cls_sim": cls_sim,
            "dino_patch_sim_mean": float(patch_sim.mean()),
            "dino_patch_sim_min": float(patch_sim.min()),
            "_patch_sim_map": patch_map,
        }


# ─── Edge F1 ─────────────────────────────────────────────────────────────────


def extract_canny_edges(
    image: Image.Image,
    low: int = 50,
    high: int = 150,
) -> np.ndarray:
    """Binary Canny edge map."""
    img_np = np.array(image)
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    return (cv2.Canny(gray, low, high) > 0).astype(np.uint8)


def compute_edge_f1(
    original: Image.Image,
    generated: Image.Image,
    tolerance_px: int = 2,
    canny_low: int = 50,
    canny_high: int = 150,
) -> dict:
    """Edge F1 with pixel tolerance (dilation-based)."""
    generated = generated.resize(original.size, Image.BILINEAR)

    edges_orig = extract_canny_edges(original, canny_low, canny_high)
    edges_gen = extract_canny_edges(generated, canny_low, canny_high)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * tolerance_px + 1, 2 * tolerance_px + 1),
    )
    orig_dilated = cv2.dilate(edges_orig, kernel)
    gen_dilated = cv2.dilate(edges_gen, kernel)

    gen_count = edges_gen.sum()
    orig_count = edges_orig.sum()

    precision = float((edges_gen * orig_dilated).sum() / gen_count) if gen_count > 0 else 1.0
    recall = float((edges_orig * gen_dilated).sum() / orig_count) if orig_count > 0 else 1.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Edge density: fraction of pixels that are edges in the original
    total_pixels = edges_orig.shape[0] * edges_orig.shape[1]
    edge_density = float(orig_count / total_pixels) if total_pixels > 0 else 0.0

    return {
        "edge_f1": f1,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_density": edge_density,
    }


# ─── SSIM on Edge Maps ───────────────────────────────────────────────────────


def compute_ssim_edges(
    original: Image.Image,
    generated: Image.Image,
    canny_low: int = 50,
    canny_high: int = 150,
) -> dict:
    """SSIM on Canny edge maps (no color confound)."""
    from scipy.ndimage import uniform_filter

    generated = generated.resize(original.size, Image.BILINEAR)

    e1 = extract_canny_edges(original, canny_low, canny_high).astype(np.float64)
    e2 = extract_canny_edges(generated, canny_low, canny_high).astype(np.float64)

    C1, C2, ws = 0.01 ** 2, 0.03 ** 2, 11

    mu1 = uniform_filter(e1, size=ws)
    mu2 = uniform_filter(e2, size=ws)
    sigma1_sq = uniform_filter(e1 ** 2, size=ws) - mu1 ** 2
    sigma2_sq = uniform_filter(e2 ** 2, size=ws) - mu2 ** 2
    sigma12 = uniform_filter(e1 * e2, size=ws) - mu1 * mu2

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / den
    pad = ws // 2
    ssim_val = float(ssim_map[pad:-pad, pad:-pad].mean())

    return {"ssim_edge": ssim_val}


# ─── Quality Gate ─────────────────────────────────────────────────────────────


class QualityGate:
    """Quality gate that rejects hallucinated generations during the pipeline.

    Loads DINOv2 once and runs all three metrics on each (original, generated) pair.
    Returns pass/fail based on configurable thresholds.

    Args:
        dino_cls_threshold: min DINOv2 CLS cosine similarity (default: 0.5)
        dino_patch_threshold: min DINOv2 mean patch similarity (default: 0.4)
        edge_f1_threshold: min edge F1 score (default: 0.3)
        device: torch device
    """

    def __init__(
        self,
        dino_cls_threshold: float = 0.5,
        dino_patch_threshold: float = 0.4,
        edge_f1_threshold: float = 0.3,
        canny_low: int = 50,
        canny_high: int = 150,
        edge_tolerance: int = 2,
        device: str = "cuda",
    ):
        self.dino_cls_threshold = dino_cls_threshold
        self.dino_patch_threshold = dino_patch_threshold
        self.edge_f1_threshold = edge_f1_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.edge_tolerance = edge_tolerance

        print(f"Loading quality gate (DINOv2 + Edge F1 + SSIM-edge)...")
        print(f"  Thresholds: dino_cls>={dino_cls_threshold}, "
              f"dino_patch>={dino_patch_threshold}, "
              f"edge_f1>={edge_f1_threshold}")
        self.dino = DINOv2Similarity(device=device)

        # Track stats
        self.total_checked = 0
        self.total_passed = 0
        self.total_rejected = 0

    def check(self, original: Image.Image, generated: Image.Image) -> tuple:
        """Check if a generated image passes the quality gate.

        Returns:
            (passed: bool, metrics: dict)
        """
        metrics = {}

        # DINOv2
        dino_result = self.dino.compute(original, generated)
        metrics.update({k: v for k, v in dino_result.items() if not k.startswith("_")})

        # Edge F1
        metrics.update(compute_edge_f1(
            original, generated,
            tolerance_px=self.edge_tolerance,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
        ))

        # SSIM on edges
        metrics.update(compute_ssim_edges(
            original, generated,
            canny_low=self.canny_low,
            canny_high=self.canny_high,
        ))

        # Check thresholds
        # Skip edge_f1 check when original has very few edges (< 1% edge pixels)
        # e.g. water/river scenes with no landmarks
        edge_sparse = metrics.get("edge_density", 1.0) < 0.01
        edge_ok = edge_sparse or metrics["edge_f1"] >= self.edge_f1_threshold
        passed = (
            metrics["dino_cls_sim"] >= self.dino_cls_threshold
            and metrics["dino_patch_sim_mean"] >= self.dino_patch_threshold
            and edge_ok
        )

        self.total_checked += 1
        if passed:
            self.total_passed += 1
        else:
            self.total_rejected += 1

        return passed, metrics

    def summary(self) -> str:
        """Return a summary string of gate statistics."""
        if self.total_checked == 0:
            return "Quality gate: no images checked"
        reject_rate = self.total_rejected / self.total_checked * 100
        return (
            f"Quality gate: {self.total_passed}/{self.total_checked} passed "
            f"({self.total_rejected} rejected, {reject_rate:.1f}% reject rate)"
        )


def add_quality_gate_args(parser):
    """Add quality gate CLI arguments to an argparse parser."""
    group = parser.add_argument_group("Quality gate (reject hallucinated images)")
    group.add_argument(
        "--quality-gate",
        action="store_true",
        help="Enable quality gate to reject structurally inconsistent generations",
    )
    group.add_argument(
        "--qg-dino-cls",
        type=float,
        default=0.5,
        help="Quality gate: min DINOv2 CLS similarity (default: 0.5)",
    )
    group.add_argument(
        "--qg-dino-patch",
        type=float,
        default=0.4,
        help="Quality gate: min DINOv2 mean patch similarity (default: 0.4)",
    )
    group.add_argument(
        "--qg-edge-f1",
        type=float,
        default=0.3,
        help="Quality gate: min edge F1 score (default: 0.3)",
    )
    group.add_argument(
        "--qg-max-retries",
        type=int,
        default=3,
        help="Quality gate: max retries per image before skipping (default: 3)",
    )


def create_quality_gate_from_args(args) -> tuple:
    """Create QualityGate from parsed args. Returns (gate_or_None, max_retries)."""
    if not getattr(args, "quality_gate", False):
        return None, 0
    gate = QualityGate(
        dino_cls_threshold=args.qg_dino_cls,
        dino_patch_threshold=args.qg_dino_patch,
        edge_f1_threshold=args.qg_edge_f1,
        device=getattr(args, "device", "cuda"),
    )
    return gate, args.qg_max_retries
