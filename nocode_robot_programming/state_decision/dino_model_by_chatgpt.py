import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

import warnings
# silence the specific xFormers-not-available notices from DINOv2
warnings.filterwarnings(
    "ignore",
    message="xFormers is not available",
    category=UserWarning,
)

class DINOProtoPresence:
    """
    Tiny 'peg present?' detector using a single DINO prototype.

    Train:
      - Extract DINO patch tokens for all images.
      - From positive (peg-present) images, collect the top-K patch tokens (per image) by a crude seed,
        then average -> prototype vector p+ (L2-normalized).
      - Optionally compute a negative prototype p- from negatives' top-K (helps calibration).

    Predict:
      - Cosine map: s_i = cos(patch_i, p+).
      - Score = max_i s_i  (translation jitter robust).
      - Threshold via a percentile of positive train scores; else return argmax( [max s_i, 1 - max s_i] ).

    Visualization:
      - Return the cosine map (M = Gh*Gw) reshaped to [Gh,Gw] and upsampled to image size.
    """

    def __init__(
        self,
        dino_variant: str = "dinov2_vits14",
        input_size: int = 224,
        batch_size: int = 64,
        pos_topk_frac: float = 0.02,    # keep top 2% patches per positive image when building prototype
        percentile_keep: float = 0.10,  # open-set / decision threshold from positives
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('facebookresearch/dinov2', dino_variant).to(self.device).eval()

        self.input_size = int(input_size)
        self.batch_size = int(batch_size)
        self.pos_topk_frac = float(pos_topk_frac)
        self.percentile_keep = float(percentile_keep)

        self.proto_pos: Optional[torch.Tensor] = None  # [D]
        self.proto_neg: Optional[torch.Tensor] = None  # [D] (optional)
        self.thr_pos: Optional[float] = None
        self.y_cls: List[str] = ["peg_present", "peg_absent"]

    def __str__(self):
        return str(self.__class__.__name__)

    def train(self, X: torch.Tensor, y: torch.Tensor, y_cls: Optional[List[str]] = None) -> None:
        """
        X: [N,H,W] float32 in [0,1] (grayscale or RGB; grayscale preferred)
        y: [N] int {0: peg present, 1: peg absent}
        """
        assert X.ndim == 3, "X must be [N,H,W]"
        y = y.long().to(self.device)
        if y_cls: self.y_cls = list(y_cls)

        # 1) Patch tokens
        P = self._all_patch_feats(X)       # [N, M, D], L2-normalized
        N, M, D = P.shape
        k = max(1, int(M * self.pos_topk_frac))

        # 2) Build positive prototype from top-k patches per positive image
        with torch.no_grad():
            pos_mask = (y == 0)
            P_pos = P[pos_mask]                                   # [Npos, M, D]
            # seed: per-image crude center (mean) then take top-k by dot to that seed
            mu_seed = F.normalize(P_pos.mean(dim=(0,1)), dim=0)   # [D]
            scores = torch.einsum('nmd,d->nm', P_pos, mu_seed)    # [Npos, M]
            _, idx = torch.topk(scores, k, dim=1, largest=True)
            P_top = torch.gather(P_pos, 1, idx.unsqueeze(-1).expand(-1,-1,D))  # [Npos, k, D]
            proto_pos = F.normalize(P_top.mean(dim=(0,1)), dim=0)              # [D]
            self.proto_pos = proto_pos

            # Optional: negative prototype for sanity / analysis
            neg_mask = (y == 1)
            if neg_mask.any():
                P_neg = P[neg_mask]
                # take patches with *lowest* similarity to proto_pos as characteristic background
                s_neg = torch.einsum('nmd,d->nm', P_neg, proto_pos)            # [Nneg, M]
                _, idx_lo = torch.topk(-s_neg, k, dim=1, largest=True)
                Pn = torch.gather(P_neg, 1, idx_lo.unsqueeze(-1).expand(-1,-1,D))
                self.proto_neg = F.normalize(Pn.mean(dim=(0,1)), dim=0)        # [D]

        # 3) Threshold from positive train scores
        with torch.no_grad():
            s_all = self._max_score_batch(P)                                   # [N]
            s_pos = s_all[pos_mask].detach().float().cpu()
            self.thr_pos = self._percentile(s_pos, self.percentile_keep)       # float

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> Tuple[str, torch.Tensor]:
        """
        image: [H,W] float32 in [0,1] (or [3,H,W]); returns (label, heatmap_up)
        heatmap_up: [1, H, W] tensor in [0,1], cosine similarity to prototype
        """
        assert self.proto_pos is not None, "Call train() first"
        P, Gh, Gw = self._single_patch_feats(image)        # [M,D], grid size
        s = P @ self.proto_pos                              # [M]
        s = s.clamp(-1, 1)                                  # cosine
        score = s.max().item()

        # decision
        if score < (self.thr_pos if self.thr_pos is not None else 0.0):
            label = self.y_cls[1]  # peg_absent
        else:
            label = self.y_cls[0]  # peg_present

        # heatmap (normalize to [0,1] for viz), upsample to image size
        hmap = (s - s.min()) / (s.max() - s.min() + 1e-8)  # [M] -> [0,1]
        hmap = hmap.view(1, 1, Gh, Gw)
        H, W = image.shape[-2], image.shape[-1]
        hmap_up = torch.nn.functional.interpolate(hmap, size=(H, W), mode="bilinear", align_corners=False)
        return label #, hmap_up.squeeze(0)  # [1,H,W]

    # -------- internals --------

    @torch.inference_mode()
    def _all_patch_feats(self, X: torch.Tensor) -> torch.Tensor:
        X = self._prep_batch(X.to(self.device))            # [N,3,S,S]
        feats = []
        for i in range(0, X.size(0), self.batch_size):
            xb = X[i:i+self.batch_size]
            out = self.model.forward_features(xb)
            P = out["x_norm_patchtokens"].float()          # [B,M,D]
            feats.append(F.normalize(P, dim=-1))
        return torch.cat(feats, dim=0)

    @torch.inference_mode()
    def _single_patch_feats(self, x: torch.Tensor) -> Tuple[torch.Tensor,int,int]:
        if x.ndim == 2:
            xb = self._prep_batch(x.unsqueeze(0))          # [1,3,S,S]
        elif x.ndim == 3 and x.size(0) == 1:
            xb = self._prep_batch(x)                       # [1,3,S,S]
        elif x.ndim == 3 and x.size(0) == 3:
            xb = self._prep_batch(x.unsqueeze(0))          # [1,3,S,S] (already 3ch)
        else:
            raise ValueError("image should be [H,W], [1,H,W], or [3,H,W]")
        out = self.model.forward_features(xb)
        P = out["x_norm_patchtokens"].float()[0]           # [M,D]
        P = F.normalize(P, dim=-1)
        # grid size
        patch = self.model.patch_embed.patch_size
        Gh = xb.shape[-2] // (patch if isinstance(patch, int) else patch[0])
        Gw = xb.shape[-1] // (patch if isinstance(patch, int) else patch[1])
        return P, Gh, Gw

    def _prep_batch(self, X: torch.Tensor) -> torch.Tensor:
        # Accepts [N,H,W] grayscale OR [N,3,H,W]; returns [N,3,S,S] ImageNet-normalized
        if X.ndim == 3:
            X = X.unsqueeze(1)
        if X.size(1) == 1:
            X = X.repeat(1, 3, 1, 1)
        X = torch.nn.functional.interpolate(
            X, size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False
        )
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
        return (X - mean) / std

    @staticmethod
    def _percentile(scores_cpu: torch.Tensor, q: float) -> float:
        k = max(0, min(len(scores_cpu)-1, int(round(q * (len(scores_cpu)-1)))))
        vals, _ = torch.sort(scores_cpu)
        return float(vals[k].item())

    @torch.inference_mode()
    def _max_score_batch(self, P: torch.Tensor) -> torch.Tensor:
        # P: [N,M,D]
        s = torch.einsum('nmd,d->nm', P, self.proto_pos)  # [N,M]
        s = s.clamp(-1, 1)
        return s.max(dim=1).values                        # [N]

    def predict_many(self, X: torch.Tensor) -> List[str]:
        return [self.predict(x) for x in X]
