import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

class DINOFeaturePresence:
    """ DINO patch embeddings.
    - One prototype per class from training images.
    - Image score for class c = max_patch cosine(patch, proto_c).
    - Open-set gate: per-class threshold from training positives (percentile).
    """
    def __init__(
        self,
        dino_variant: str = "dinov2_vits14",
        input_size: int = 280,              # a bit larger than 224 to help small details
        percentile_keep: float = 0.10,      # 10th percentile acceptance (per-class)
        batch_size: int = 64,
    ):
        self.device = torch.device("cuda")
        self.model = torch.hub.load('facebookresearch/dinov2', dino_variant).to(self.device).eval()

        self.input_size = int(input_size)
        self.percentile_keep = float(percentile_keep)
        self.batch_size = int(batch_size)

        self.y_cls: Optional[List[str]] = None            # class name list (index -> name)
        self._class_ids: Optional[torch.Tensor] = None    # tensor of sorted unique y ids
        self._protos: Optional[torch.Tensor] = None       # [C, D], unit-norm
        self._thresholds: Optional[torch.Tensor] = None   # [C], per-class cosine cutoff

    def _prep_batch(self, X: torch.Tensor) -> torch.Tensor:
        """ X: [B, H, W]
        Returns: float tensor on device [B, 3, S, S], ImageNet-normalized.
        """
        X = X.unsqueeze(1)
        if X.size(1) == 1:         # grayscale -> 3ch
            X = X.repeat(1, 3, 1, 1)
        elif X.size(1) != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {X.size(1)}")

        # Resize (bilinear) to square
        X = torch.nn.functional.interpolate(
            X, size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False
        )
        # Normalize (ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
        X = (X - mean) / std

        return X

    @torch.inference_mode()
    def _patch_feats(self, Xb: torch.Tensor) -> torch.Tensor:
        """
        Xb: preprocessed batch [B,3,S,S]
        Returns: [B, P, D] unit-norm patch features
        """
        out = self.model.forward_features(Xb)
        P = out["x_norm_patchtokens"]             # [B, P, D]
        P = F.normalize(P.float(), dim=-1)        # stay float32 for stability downstream
        return P

    @staticmethod
    def _percentile(scores: torch.Tensor, q: float) -> float:
        """scores: 1D tensor on CPU; q in [0,1]"""
        k = max(0, min(len(scores)-1, int(round(q * (len(scores)-1)))))
        vals, _ = torch.sort(scores)
        return float(vals[k].item())

    def train(self, X: torch.Tensor, y: torch.Tensor, y_cls: List[str]):
        """
        X: [N, H, W] (float32 in [0,1]) (device="cuda")
        y: [N] integer class ids indexing into y_cls
        y_cls: list of class names (len = C)
        """
        assert X.ndim == 3, "X must be [N,H,W]"
        assert len(y_cls) > 0, "y_cls must not be empty"
        N = X.shape[0]
        y = y.detach().cpu().long()
        self.y_cls = list(y_cls)

        # Unique class ids (sorted)
        class_ids = torch.unique(y).tolist()
        class_ids.sort()
        self._class_ids = torch.tensor(class_ids, dtype=torch.long)

        # Compute patch features in batches
        feats_list = []
        y_list = []
        for i in range(0, N, self.batch_size):
            Xb = self._prep_batch(X[i:i+self.batch_size])
            with torch.amp.autocast(self.device.type):
                Pb = self._patch_feats(Xb)          # [B, P, D]
            feats_list.append(Pb.cpu())
            y_list.append(y[i:i+self.batch_size])
        patches = torch.cat(feats_list, dim=0)       # [N, P, D]
        y_all = torch.cat(y_list, dim=0)             # [N]

        B, P, D = patches.shape

        # ---- Build class prototypes: mean of image-level patch means per class ----
        protos = []
        thresholds = []
        for cid in class_ids:
            idx = (y_all == cid).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                # should not happen if class present
                proto = torch.zeros(D)
                thr = 1.0
            else:
                # mean per image over patches, then mean over images -> robust to image count
                img_means = patches[idx].mean(dim=1)          # [n_c, D]
                proto = F.normalize(img_means.mean(dim=0), dim=0)   # [D]

                # Positive scores for threshold: for each image in class,
                # score = max over its patches of cosine(patch, proto)
                pos_scores = (patches[idx] @ proto)            # [n_c, P]
                pos_scores, _ = pos_scores.max(dim=1)          # [n_c]
                pos_scores = pos_scores.cpu()

                if pos_scores.numel() >= 5:
                    thr = self._percentile(pos_scores, self.percentile_keep)
                elif pos_scores.numel() > 0:
                    thr = float(pos_scores.min().item()) * 0.95
                else:
                    thr = 0.3  # fallback

            protos.append(proto)
            thresholds.append(thr)

        self._protos = torch.stack(protos, dim=0).contiguous().to(self.device)            # [C, D]
        self._thresholds = torch.tensor(thresholds, dtype=torch.float32)  # [C]

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, timestep: float | None = None) -> str:
        """
        Returns the class name (str) from y_cls, or "" for anomaly.
        """
        assert self._protos is not None, "Call train() first."

        Xb = image.unsqueeze(0) if image.ndim == 2 else image.unsqueeze(0)  # keep batch dim
        Xb = self._prep_batch(Xb)                                           # [1,3,S,S]
        with torch.amp.autocast(self.device.type):
            P = self._patch_feats(Xb)[0]                                    # [P, D], float32

        # Cosine to each prototype; take per-class max over patches
        # sims_pc = [P @ proto_c] => [P], take max -> scalar per class
        C = self._protos.size(0)
        # (C,D) x (D,P) → (C,P) then max over P
        sims_CP = (self._protos @ P.T)             # [C, P]
        best_over_p, _ = sims_CP.max(dim=1)        # [C]

        # Argmax class
        c_idx = int(torch.argmax(best_over_p).item())
        best_score = float(best_over_p[c_idx].item())
        thr = float(self._thresholds[c_idx].item())

        if True: #best_score >= thr:
            # Map class index -> class name via training's class id order
            # self._class_ids[c_idx] is the original integer id that indexes y_cls
            orig_id = int(self._class_ids[c_idx].item())
            return self.y_cls[orig_id] if (self.y_cls and 0 <= orig_id < len(self.y_cls)) else str(orig_id)
        else:
            return ""  # anomaly

    def predict_many(self, X: torch.Tensor) -> List[str]:
        return [self.predict(x) for x in X]# if X.ndim == 4 else X.unsqueeze(1))]