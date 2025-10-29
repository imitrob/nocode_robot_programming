import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

import warnings
# silence the specific xFormers-not-available notices from DINOv2
warnings.filterwarnings(
    "ignore",
    message="xFormers is not available",
    category=UserWarning,
)

class DINOFeaturePresence:
    """ DINO-based classifier with cosine prototypes over patch tokens, see tutorial: `dino_tutorial.ipynb`
    
    C: int number of classes
    D: int number of features (embedding)
    N: int number of training samples
    H: int Height pixels
    W: int Width pixels
    """
    def __init__(
        self,
        dino_variant: str = "dinov2_vits14",
        input_size: int = 224, # pixels width/height
        batch_size: int = 64,
        percentile_keep: Optional[float] = None,   # e.g., 0.10 to enable open-set gating
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('facebookresearch/dinov2', dino_variant).to(self.device).eval()

        self.input_size = int(input_size)
        self.batch_size = int(batch_size)
        self.percentile_keep = percentile_keep

        # learned with train()
        self.y_cls: List[str] = []
        self.prototypes: Optional[torch.Tensor] = None   # [C, D] (L2-normalized)
        self.thresholds: Optional[torch.Tensor] = None   # [C] per-class open-set thresholds (optional)

    def __str__(self):
        return str(self.__class__.__name__)

    def train(self, X: torch.Tensor, y: torch.Tensor, y_cls: List[str]) -> None:
        """
        X: [N, H, W] float32 in [0,1] (any device; will be moved)
        y: [N] int64 class ids
        y_cls: list of class names, len C
        """
        assert X.ndim == 3, "X must be [N,H,W]"
        self.y_cls = list(y_cls)
        C = len(y_cls)

        # 1) Compute patch features for all images
        P = self._all_patch_feats(X)  # [N, M, D], L2-normalized
        N, M, D = P.shape

        # 2) Pool image -> single vector (default: mean over patches)
        G = self._pool_patches(P)  # [N, D], L2-normalized

        # 3) Build class prototypes as mean of pooled features, then L2-normalize
        prototypes = torch.zeros(C, D, device=self.device)
        for c in range(C):
            mask = (y == c)
            assert mask.any(), f"No samples for class id {c}"
            mu = G[mask].mean(dim=0)
            prototypes[c] = F.normalize(mu, dim=0)
        self.prototypes = prototypes                   # [C, D]

        # 4) Optional: per-class open-set thresholds from training scores
        if self.percentile_keep is not None:
            with torch.no_grad():
                logits = self._score_logits(G, self.prototypes)  # [N, C]
                pos_scores = []                                  # list of tensors (per class)
                for c in range(C):
                    pos_scores.append(logits[y == c, c].detach().float().cpu())
            thresholds = []
            for c in range(C):
                s = pos_scores[c].sort().values
                k = max(0, min(len(s)-1, int(round(self.percentile_keep * (len(s)-1)))))
                thresholds.append(s[k].item())
            self.thresholds = torch.tensor(thresholds, device=self.device)

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> str:
        """
        image: [H, W] float32 in [0,1]
        returns: predicted class name (or 'unknown' if gated)
        """
        assert self.prototypes is not None, "Call train() first"
        p = self._single_patch_feats(image)  # [M, D]
        g = self._pool_patches(p.unsqueeze(0))[0]  # [D]
        logits = self._score_logits(g.unsqueeze(0), self.prototypes)[0]  # [C]
        c = int(torch.argmax(logits).item())

        if self.percentile_keep is not None and self.thresholds is not None: # enabled
            if logits[c] < self.thresholds[c]:
                return "unknown"
        
        return self.y_cls[c]

    def predict_many(self, X: torch.Tensor) -> List[str]:
        return [self.predict(x) for x in X]

    def _pool_patches(self, P: torch.Tensor) -> torch.Tensor:
        """ Default pooling: L2-normalized mean over patches. This simplifies things a lot, place for improvement - override this method. 
        P: [N, M, D] or [1, M, D]
        Returns: [N, D]
        """
        G = P.mean(dim=1)  # [N, D]
        return F.normalize(G, dim=-1)

    def _score_logits(self, G: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Cosine logits (no temperature by default).
        G: [N, D] (L2-normalized)
        prototypes: [C, D] (L2-normalized)
        Returns: [N, C]
        """
        return G @ prototypes.T

    @torch.inference_mode()
    def _all_patch_feats(self, X: torch.Tensor) -> torch.Tensor:
        """ Used DINO model here in bachted patches.

        X: [N, H, W] float32 in [0,1]
        Returns: [N, M, D] (L2-normalized)
        """
        X = X.to(self.device, non_blocking=True)
        feats = []
        for i in range(0, X.size(0), self.batch_size):
            xb = self._prep_batch(X[i:i+self.batch_size])  # [B,3,S,S]
            out = self.model.forward_features(xb)
            P = out["x_norm_patchtokens"].float()  # [B, M, D]
            feats.append(F.normalize(P, dim=-1))
        return torch.cat(feats, dim=0)

    @torch.inference_mode()
    def _single_patch_feats(self, x: torch.Tensor) -> torch.Tensor:
        """ Used DINO model here.

        x: [H, W] float32 in [0,1]
        Returns: [M, D]
        """
        xb = self._prep_batch(x.unsqueeze(0))
        out = self.model.forward_features(xb)
        P = out["x_norm_patchtokens"].float()[0]  # [M, D]
        return F.normalize(P, dim=-1)

    def _prep_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, H, W] float32 in [0,1]
        Returns: [B, 3, S, S] ImageNet-normalized
        """
        if X.ndim == 3:  # [B,H,W], grayscale
            X = X.unsqueeze(1).repeat(1, 3, 1, 1)
        elif X.ndim == 4 and X.size(1) == 1:
            X = X.repeat(1, 3, 1, 1)
        elif X.ndim == 4 and X.size(1) == 3:
            pass
        else:
            raise ValueError("Expected [B,H,W] or [B,1,H,W] or [B,3,H,W]")

        X = torch.nn.functional.interpolate(
            X, size=(self.input_size, self.input_size),
            mode="bicubic", align_corners=False
        )
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
        return (X - mean) / std
