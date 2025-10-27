import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class DINOFeaturePresence:
    """ DINO-based classifier with cosine prototypes over patch tokens.
        Extension-ready: override hooks (_pool_patches, _score_logits, etc.) in subclasses.

    - Trains by computing per-class prototypes from training images.
    - Predicts by pooling image patch features, then cosine to class prototypes.
    - Optional open-set gating via per-class percentile threshold on train scores.
    """

    def __init__(
        self,
        dino_variant: str = "dinov2_vits14",
        input_size: int = 224,
        batch_size: int = 64,
        percentile_keep: Optional[float] = None,   # e.g., 0.10 to enable open-set gating
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('facebookresearch/dinov2', dino_variant).to(self.device).eval()

        self.input_size = int(input_size)
        self.batch_size = int(batch_size)
        self.percentile_keep = percentile_keep

        # learned / derived after train()
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
        P = self._all_patch_feats(X)                    # [N, M, D], L2-normalized
        N, M, D = P.shape

        # 2) Pool image -> single vector (default: mean over patches)
        G = self._pool_patches(P)                      # [N, D], L2-normalized

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
        p = self._single_patch_feats(image)            # [M, D]
        g = self._pool_patches(p.unsqueeze(0))[0]      # [D]
        logits = self._score_logits(g.unsqueeze(0), self.prototypes)[0]  # [C]
        c = int(torch.argmax(logits).item())
        # if self.thresholds is not None and logits[c] < self.thresholds[c]:
        #     return "unknown"
        return self.y_cls[c]

    def predict_many(self, X: torch.Tensor) -> List[str]:
        return [self.predict(x) for x in X]

    def _pool_patches(self, P: torch.Tensor) -> torch.Tensor:
        """
        Default pooling: L2-normalized mean over patches.
        P: [N, M, D] or [1, M, D]
        Returns: [N, D]
        """
        G = P.mean(dim=1)                      # [N, D]
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
        """
        Batched patch embeddings.
        X: [N, H, W] float32 in [0,1] (CPU or GPU)
        Returns: [N, M, D] (L2-normalized)
        """
        X = X.to(self.device, non_blocking=True)
        feats = []
        for i in range(0, X.size(0), self.batch_size):
            xb = self._prep_batch(X[i:i+self.batch_size])         # [B,3,S,S]
            out = self.model.forward_features(xb)
            P = out["x_norm_patchtokens"].float()                 # [B, M, D]
            feats.append(F.normalize(P, dim=-1))
        return torch.cat(feats, dim=0)

    @torch.inference_mode()
    def _single_patch_feats(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [H, W] float32 in [0,1]
        Returns: [M, D]
        """
        xb = self._prep_batch(x.unsqueeze(0))
        out = self.model.forward_features(xb)
        P = out["x_norm_patchtokens"].float()[0]                  # [M, D]
        return F.normalize(P, dim=-1)

    def _prep_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, H, W] float32 in [0,1]
        Returns: [B, 3, S, S] ImageNet-normalized
        """
        if X.ndim == 3:      # [B,H,W], grayscale
            X = X.unsqueeze(1).repeat(1, 3, 1, 1)
        elif X.ndim == 4 and X.size(1) == 1:
            X = X.repeat(1, 3, 1, 1)
        elif X.ndim == 4 and X.size(1) == 3:
            pass
        else:
            raise ValueError("Expected [B,H,W] or [B,1,H,W] or [B,3,H,W]")

        X = torch.nn.functional.interpolate(
            X, size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False
        )
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
        return (X - mean) / std


import torch
import torch.nn.functional as F
from typing import Optional, List

class DINOWithMIL(DINOFeaturePresence):
    """
    Multiple-Instance Learning (MIL) over DINO patch tokens.

    - Learns a small attention head to pool patches into an image embedding.
    - Classifier is cosine to learnable class weights (W_cls).
    - Falls back to mean pooling if head isn't initialized (e.g., before train()).
    """

    def __init__(
        self,
        *args,
        att_hidden: int = 128,
        dropout_p: float = 0.1,
        topk_frac: Optional[float] = 0.1,      # None for full attention sum
        lr: float = 7e-5,
        weight_decay: float = 1e-3,
        epochs: int = 200,
        patience: int = 20,
        seed: Optional[int] = 125,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.att_hidden = int(att_hidden)
        self.dropout_p = float(dropout_p)
        self.topk_frac = topk_frac
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.seed = seed

        # MIL params (created lazily in train() when D and C are known)
        self.W_att: Optional[torch.Tensor] = None   # [H, D]
        self.b_att: Optional[torch.Tensor] = None   # [H]
        self.v_att: Optional[torch.Tensor] = None   # [H]
        self.W_cls: Optional[torch.Tensor] = None   # [C, D]

        self._dropout = torch.nn.Dropout(self.dropout_p)

    # ---------- Public API (override) ----------

    def train(self, X: torch.Tensor, y: torch.Tensor, y_cls: List[str]) -> None:
        """
        Supervised MIL training of the attention head + class weights.
        X: [N, H, W] float32 in [0,1]
        y: [N] int64
        """
        assert X.ndim == 3, "X must be [N,H,W]"
        self.y_cls = list(y_cls)
        C = len(y_cls)
        y = y.to(self.device).long()   # CE expects int64 class indices on the same device

        # Reproducibility
        if self.seed is not None:
            import random, numpy as np
            random.seed(self.seed); np.random.seed(self.seed)
            torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 1) Extract all patch features (frozen DINO)
        P = self._all_patch_feats(X)                         # [N, M, D]
        N, M, D = P.shape

        # 2) Initialize MIL head + class weights (cosine)
        self._init_head(D, C)

        params = [self.W_att, self.b_att, self.v_att, self.W_cls]
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Simple stratified split (20% val) on CPU indices
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        idx_all = torch.arange(N, device='cpu').numpy()
        y_cpu = y.detach().cpu().numpy()
        tr_idx, va_idx = next(sss.split(idx_all, y_cpu))
        tr_idx = torch.tensor(tr_idx, device=P.device)
        va_idx = torch.tensor(va_idx, device=P.device)

        best_state, best_val, bad = None, -1.0, 0

        losses = []
        tau_start, tau_end = 0.7, 0.18  # explore -> focus
        for ep in range(self.epochs):
            t = ep / max(1, self.epochs-1)
            self.tau = tau_end + 0.5*(tau_start - tau_end)*(1 + torch.cos(torch.tensor(t*3.1415926535))).item()

            # ---- train ----
            self._dropout.train()
            opt.zero_grad(set_to_none=True)

            P_tr = P[tr_idx]                 # [Bt,M,D]
            y_tr = y[tr_idx]                 # [Bt]
            alpha_tr = self._attention_weights(P_tr)               # [Bt,M]
            logits_tr = self._class_logits(P_tr, alpha_tr)         # [Bt,C]
            loss = loss_fn(logits_tr, y_tr)
            losses.append(float(loss.item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

            # ---- validate ----
            self._dropout.eval()
            with torch.inference_mode():
                P_va = P[va_idx]
                y_va = y[va_idx]
                a_va = self._attention_weights(P_va)
                logits_va = self._class_logits(P_va, a_va)
                acc = (logits_va.argmax(1) == y_va).float().mean().item()

            if acc > best_val:
                best_val = acc
                best_state = {k: p.detach().clone() for k, p in zip(
                    ["W_att","b_att","v_att","W_cls"], params)}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        # Load best params
        if best_state is not None:
            self.W_att.data.copy_(best_state["W_att"])
            self.b_att.data.copy_(best_state["b_att"])
            self.v_att.data.copy_(best_state["v_att"])
            self.W_cls.data.copy_(best_state["W_cls"])

        # Optional open-set thresholds (on all train images)
        if self.percentile_keep is not None:
            with torch.inference_mode():
                a_all = self._attention_weights(P)
                logits_all = self._class_logits(P, a_all)          # [N,C]
            thresholds = []
            for c in range(C):
                s = logits_all[y == c, c].detach().float().cpu().sort().values
                k = max(0, min(len(s)-1, int(round(self.percentile_keep * (len(s)-1)))))
                thresholds.append(s[k].item())
            self.thresholds = torch.tensor(thresholds, device=self.device)

        import matplotlib.pyplot as plt
        plt.plot(losses)

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> str:
        """
        image: [H, W] float32 in [0,1]
        """
        assert self.W_att is not None and self.W_cls is not None, "Call train() first"
        p = self._single_patch_feats(image)                        # [M,D]
        a = self._attention_weights(p.unsqueeze(0))[0]             # [M]
        logits = self._class_logits(p.unsqueeze(0), a.unsqueeze(0))[0]  # [C]
        c = int(torch.argmax(logits).item())
        if self.thresholds is not None and logits[c] < self.thresholds[c]:
            return "unknown"
        return self.y_cls[c]

    # ---------- Hooks (override base pooling & scoring) ----------

    def _pool_patches(self, P: torch.Tensor) -> torch.Tensor:
        """
        MIL pooling using learned attention.
        P: [N,M,D] or [1,M,D]
        Returns normalized [N,D].
        """
        if self.W_att is None:                    # before train() fallback
            G = P.mean(dim=1)
            return F.normalize(G, dim=-1)
        alpha = self._attention_weights(P)        # [N,M]
        # cosine patch scores to class weights is used in _class_logits; for pooled
        # image embedding (class-agnostic), do weighted mean over patches:
        G = torch.einsum('nm,nmd->nd', alpha, P)  # [N,D]
        return F.normalize(G, dim=-1)

    def _score_logits(self, G: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        (Not used directly in MIL; kept for API compatibility.)
        """
        return super()._score_logits(G, prototypes)

    # ---------- MIL internals ----------

    def _init_head(self, D: int, C: int) -> None:
        """
        Create/initialize attention and class-weight params given dims.
        """
        device = self.device
        # Attention
        W_att = torch.empty(self.att_hidden, D, device=device)
        torch.nn.init.xavier_uniform_(W_att)
        b_att = torch.zeros(self.att_hidden, device=device)
        v_att = 0.01 * torch.randn(self.att_hidden, device=device)

        # Class weights (cosine classifier); start from mean-pooled prototypes for stability
        # Build quick mean-prototype from a tiny dummy tensor (will be updated in training anyway).
        # Here we just random-normalize:
        W_cls = torch.randn(C, D, device=device)
        W_cls = F.normalize(W_cls, dim=1)

        self.W_att = torch.nn.Parameter(W_att, requires_grad=True)
        self.b_att = torch.nn.Parameter(b_att, requires_grad=True)
        self.v_att = torch.nn.Parameter(v_att, requires_grad=True)
        self.W_cls = torch.nn.Parameter(W_cls, requires_grad=True)

    def entmax15(self, inputs: torch.Tensor, dim: int = -1, n_iter: int = 50, eps: float = 1e-6):
        """
        Entmax-1.5 (sparse softmax). Returns nonnegative weights summing to 1, with many exact zeros.
        inputs: [..., K]
        """
        # Shift for stability
        X = inputs - inputs.max(dim=dim, keepdim=True).values

        # Compute threshold tau by bisection so that sum(max(X - tau, 0)^(1/ (alpha-1))) == 1
        # For alpha=1.5, power = 2 (since 1/(alpha-1)=1/0.5=2)
        def _proj(u):
            # returns tau s.t. sum((u - tau)_+^2) = 1
            mu, _ = torch.sort(u, dim=dim, descending=True)
            cssv = (mu.cumsum(dim) - (torch.arange(mu.size(dim), device=mu.device) + 1).view(
                *((1,) * (mu.dim() - 1)), -1
            ) * mu)  # cumulative sum minus k * mu_k
            # Find rho: largest k such that mu_k - (cssv_k / k) > 0
            # For entmax15, formula simplifies if we go via bisection. We'll just do bisection:
            lo = (u.min(dim=dim, keepdim=True).values - 1.0)
            hi = u.max(dim=dim, keepdim=True).values
            for _ in range(n_iter):
                tau = (lo + hi) / 2.0
                out = torch.clamp(u - tau, min=0.0)
                f = (out.pow(2)).sum(dim=dim, keepdim=True) - 1.0
                hi = torch.where(f > 0, hi, tau)
                lo = torch.where(f > 0, tau, lo)
            tau = (lo + hi) / 2.0
            return tau

        tau = _proj(X)
        out = torch.clamp(X - tau, min=0.0)
        out = out.pow(2)  # power = 1/(alpha-1) = 2
        # normalize
        Z = out.sum(dim=dim, keepdim=True) + eps
        return out / Z


    # def _attention_weights(self, P: torch.Tensor) -> torch.Tensor:
    #     """
    #     P: [N,M,D] -> alpha: [N,M] with sum(alpha_i)=1 per image.
    #     e_i = v^T tanh(W p_i + b); alpha = softmax(e)
    #     """
    #     N, M, D = P.shape
    #     Z = torch.tanh(F.linear(P, self.W_att, self.b_att))  # [N,M,H]
    #     Z = self._dropout(Z)
    #     e = F.linear(Z, self.v_att.unsqueeze(0))            # [N,M,1]
    #     e = e.squeeze(-1)                                   # [N,M]
    #     alpha = torch.softmax(e, dim=1)                     # [N,M]

    #     # Optional top-k weighted mean (stabilizes tiny objects)
    #     if self.topk_frac is not None and 0.0 < self.topk_frac < 1.0:
    #         k = max(1, int(M * self.topk_frac))
    #         vals, idx = torch.topk(alpha, k=k, dim=1, largest=True, sorted=False)  # [N,k]
    #         mask = torch.zeros_like(alpha).scatter_(1, idx, 1.0)                   # [N,M]
    #         # renormalize over the selected entries
    #         alpha = (alpha * mask)
    #         alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
    #     return alpha

    def _attention_weights(self, P: torch.Tensor) -> torch.Tensor:
        # P: [N,M,D] -> alpha: [N,M]
        Z = torch.tanh(F.linear(P, self.W_att, self.b_att))   # [N,M,H]
        Z = self._dropout(Z)
        e = F.linear(Z, self.v_att.unsqueeze(0)).squeeze(-1)  # [N,M]
        alpha = self.entmax15(e, dim=1)                       # sparse, sums to 1
        return alpha

    def _class_logits(self, P: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Cosine logits computed from attention-weighted pooled features.
        P: [N,M,D], alpha: [N,M]  -> logits: [N,C]
        """
        # Weighted pooled image embeddings
        G = torch.einsum('nm,nmd->nd', alpha, P)            # [N,D]
        G = F.normalize(G, dim=1)
        W = F.normalize(self.W_cls, dim=1)                  # [C,D]
        return torch.einsum('nd,cd->nc', G, W)
