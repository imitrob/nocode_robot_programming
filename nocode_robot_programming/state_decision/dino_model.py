import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
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
        self.dino_variant = dino_variant
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', dino_variant).to(self.device).eval()
        except RuntimeError: # not found in torch.hub, try huggingface
            processor = AutoImageProcessor.from_pretrained(dino_variant)
            self.model = AutoModel.from_pretrained(dino_variant).to("cuda").eval()
            ## Monkey patch
            @torch.inference_mode()
            def forward_features_dinov3_like_dinov2(model, pixel_values, bool_masked_pos=None):
                """
                Mimics dinov2.models.vision_transformer.DinoVisionTransformer.forward_features()
                but for HF Transformers DINOv3.

                Returns:
                dict with keys:
                    - x_norm_clstoken
                    - x_norm_regtokens
                    - x_norm_patchtokens
                    - x_prenorm   (NOT available in HF; set to None)
                    - masks       (maps to bool_masked_pos)
                """
                out = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos, return_dict=True)

                x_norm = out.last_hidden_state  # (B, 1 + R + P, C)
                R = int(getattr(model.config, "num_register_tokens", 0))

                return {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : 1 + R],
                    "x_norm_patchtokens": x_norm[:, 1 + R :],
                    "x_prenorm": None,          # HF DINOv3 does not expose the pre-final-LN tokens like DINOv2 does
                    "masks": bool_masked_pos,    # HF name for masked-patch positions
                }
            
            # bind to this specific model instance
            import types
            self.model.forward_features = types.MethodType(forward_features_dinov3_like_dinov2, self.model)

        self.input_size = int(input_size)
        self.batch_size = int(batch_size)
        self.percentile_keep = percentile_keep

        # learned with train()
        self.y_cls: List[str] = []
        self.prototypes: Optional[torch.Tensor] = None   # [C, D] (L2-normalized)
        self.thresholds: Optional[torch.Tensor] = None   # [C] per-class open-set thresholds (optional)

    def __str__(self):
        return f"{self.dino_variant},{self.input_size},mean"
    
    @property
    def short_name(self) -> str:
        s = self.__str__()

        TO_NICE_NAMES = { # complicated name -> nice name
            'dinov2_vits14,224,mean': 'dinov2 small mean',
            'facebook/dinov3-vits16-pretrain-lvd1689m,224,mean': 'dinov3 small mean',
            'facebook/dinov3-vitl16-pretrain-lvd1689m,224,mean': 'dinov3 large mean',
            'dinov2_vits14,224,concat': 'dinov2 small concat',
            'dinov2_vits14,224,attn,hard,mean,0.4': 'dinov2 small attn',
            'dinov2_vits14,224,MIL,H=128,e=1000': 'dinov2 small MIL',
            'SIFT': "SIFT",
            'AEGP,bin=False': 'AEGP Multiclass',
        }
        if s in TO_NICE_NAMES:
            return TO_NICE_NAMES[s]
        else:
            return s

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
        G = self._pool_patches(P)  # [N, D | M*D], L2-normalized
        D_concat = G.size(-1) # [D (mean) or M*D (concat)]

        # 3) Build class prototypes as mean of pooled features, then L2-normalize
        prototypes = torch.zeros(C, D_concat, device=self.device)
        for c in range(C):
            mask = (y == c)
            assert mask.any(), f"No samples for class id {c}"
            mu = G[mask].mean(dim=0)
            prototypes[c] = F.normalize(mu, dim=0) # adds 2% accuracy on test dataset
        self.prototypes = prototypes                   # [C, D]

        # 4) Optional: per-class open-set thresholds from training scores
        if self.percentile_keep is not None:
            with torch.no_grad():
                logits = self._score_logits(G, self.prototypes)  # [N, C]
                pos_scores = []                                  # list of tensors (per class)
                for c in range(C):
                    pos_scores.append(logits[y == c, c].detach().float().cpu())
                # pos_scores = [logits[y == c, c].detach().float().cpu() for c in range(C)]
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
        returns: predicted class name (or 'anomaly' if gated)
        """
        assert self.prototypes is not None, "Call train() first"
        p = self._single_patch_feats(image)  # [M, D]
        g = self._pool_patches(p.unsqueeze(0))[0]  # [D]
        logits = self._score_logits(g.unsqueeze(0), self.prototypes)[0]  # [C]
        c = int(torch.argmax(logits).item())

        if self.percentile_keep is not None and self.thresholds is not None: # enabled
            if logits[c] < self.thresholds[c]:
                return "anomaly"
        
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


class DINOFeaturePresenceConcat(DINOFeaturePresence):
    """ Pooling by concatenating patch embeddings.
        This preserves per-patch information at the cost of higher dimensionality.

        P: [N, M, D]  ->  G: [N, M*D] (then L2-normalized)
    """
    def __str__(self):
        return f"{self.dino_variant},{self.input_size},concat"

    def _pool_patches(self, P: torch.Tensor) -> torch.Tensor:
        """ Concatenate patch embeddings along the feature axis, then L2-normalize.

        P: [N, M, D] or [1, M, D]
        Returns: [N, M*D]
        """
        if P.ndim != 3:
            raise ValueError("Expected P to be [N, M, D]")
        N, M, D = P.shape
        G = P.reshape(N, M * D)  # concatenation of patches
        return F.normalize(G, dim=-1)  # keep cosine geometry

class DINOFeaturePresenceAttnGated(DINOFeaturePresence):
    """ Focus on high self-attention patches
    
    - Uses last-layer CLS->patch attention to mask/weight patches.
    - Works at both training and prediction for consistency.

    Args:
        attn_keep: float in (0,1], fraction of patches to keep by attention (e.g., 0.2 = top 20%).
        attn_mode: 'hard' -> drop low-attention patches; 'soft' -> attention-weighted average.
        head_reduce: 'mean' or 'max' over attention heads.
    """
    def __init__(self, *args,
                 attn_keep: float = 0.2,
                 attn_mode: str = "hard",
                 head_reduce: str = "mean",
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert 0 < attn_keep <= 1.0
        assert attn_mode in ("hard", "soft")
        assert head_reduce in ("mean", "max")
        self.attn_keep = float(attn_keep)
        self.attn_mode = attn_mode
        self.head_reduce = head_reduce

    def __str__(self):
        return f"{self.dino_variant},{self.input_size},attn,{self.attn_mode},{self.head_reduce},{self.attn_keep}"

    def train(self, X: torch.Tensor, y: torch.Tensor, y_cls: List[str]) -> None:
        assert X.ndim == 3, "X must be [N,H,W]"
        self.y_cls = list(y_cls)
        C = len(y_cls)

        # 1) Patch feats + attention weights
        P, W = self._all_patch_feats_with_attn(X)   # P:[N,M,D] (L2), W:[N,M] in [0,1]

        # 2) Attention-gated pooling
        G = self._pool_patches_with_weights(P, W)   # [N,D] (L2)

        # 3) Prototypes from gated pooled features
        D = G.size(-1)
        prototypes = torch.zeros(C, D, device=self.device)
        for c in range(C):
            mask = (y == c)
            assert mask.any(), f"No samples for class id {c}"
            mu = G[mask].mean(dim=0)
            prototypes[c] = F.normalize(mu, dim=0)
        self.prototypes = prototypes

        # 4) Optional: per-class thresholds on the *gated* logits
        if self.percentile_keep is not None:
            with torch.no_grad():
                logits = self._score_logits(G, self.prototypes)  # [N,C]
                pos_scores = [logits[y == c, c].detach().float().cpu() for c in range(C)]
            thresholds = []
            for c in range(C):
                s = pos_scores[c].sort().values
                k = max(0, min(len(s)-1, int(round(self.percentile_keep * (len(s)-1)))))
                thresholds.append(s[k].item())
            self.thresholds = torch.tensor(thresholds, device=self.device)

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> str:
        assert self.prototypes is not None, "Call train() first"
        p, w = self._single_patch_feats_with_attn(image)        # [M,D], [M]
        g = self._pool_patches_with_weights(p.unsqueeze(0), w.unsqueeze(0))[0]  # [D]
        logits = self._score_logits(g.unsqueeze(0), self.prototypes)[0]  # [C]
        c = int(torch.argmax(logits).item())
        if self.percentile_keep is not None and self.thresholds is not None:
            if logits[c] < self.thresholds[c]:
                return "anomaly"
        return self.y_cls[c]

    def _pool_patches_with_weights(self, P: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        P: [N,M,D] (L2-normalized)
        W: [N,M]   attention weights in [0,1] (not necessarily summing to 1)
        Returns: [N,D] L2-normalized pooled representation.
        """
        eps = 1e-8
        N, M, D = P.shape
        assert W.shape == (N, M)

        # determine per-sample threshold for top-p keep
        if self.attn_keep < 1.0:
            k = (W > -1)  # dummy same shape
            # compute kth value per row
            K = torch.clamp(torch.tensor((M * self.attn_keep)).round().long(), min=1, max=M)
            # top-k mask per sample
            thresh = torch.topk(W, k=K.item(), dim=1).values[:, -1]  # [N]
            keep_mask = (W >= thresh[:, None]).float()
        else:
            keep_mask = torch.ones_like(W)

        if self.attn_mode == "hard":
            weights = keep_mask
        else:  # 'soft'
            # zero-out low attention then normalize to sum=1 per sample
            weights = W * keep_mask
            weights = weights / (weights.sum(dim=1, keepdim=True) + eps)

        # For 'hard', if all-zeros (pathological), fall back to mean
        if self.attn_mode == "hard":
            zeros = (weights.sum(dim=1) < eps)  # [N]
            if zeros.any():
                weights[zeros] = 1.0
            weights = weights / (weights.sum(dim=1, keepdim=True) + eps)

        G = torch.einsum('nmd,nm->nd', P, weights)  # weighted sum/mean
        return F.normalize(G, dim=-1)

    @torch.inference_mode()
    def _all_patch_feats_with_attn(self, X: torch.Tensor):
        """
        X: [N,H,W] float32 in [0,1]
        Returns:
            P: [N,M,D] L2-normalized patch features
            W: [N,M]   attention weights in [0,1]
        """
        X = X.to(self.device, non_blocking=True)
        feats, weights = [], []
        for i in range(0, X.size(0), self.batch_size):
            xb = self._prep_batch(X[i:i+self.batch_size])  # [B,3,S,S]
            out, A = self._forward_with_attn(xb)           # out: dict, A:[B,H,T,T] or None
            P = out["x_norm_patchtokens"].float()          # [B,M,D]
            P = F.normalize(P, dim=-1)
            W = self._cls_to_patch_weights(A, P.size(1)) if A is not None else torch.ones(P.size(0), P.size(1), device=P.device)
            feats.append(P)
            weights.append(W)
        return torch.cat(feats, dim=0), torch.cat(weights, dim=0)

    @torch.inference_mode()
    def _single_patch_feats_with_attn(self, x: torch.Tensor):
        """
        x: [H,W] float32 in [0,1]
        Returns:
            P: [M,D] L2-normalized patch features
            W: [M]   attention weights in [0,1]
        """
        xb = self._prep_batch(x.unsqueeze(0))
        out, A = self._forward_with_attn(xb)               # A:[1,H,T,T] or None
        P = out["x_norm_patchtokens"].float()[0]           # [M,D]
        P = F.normalize(P, dim=-1)
        if A is None:
            W = torch.ones(P.size(0), device=P.device)
        else:
            W = self._cls_to_patch_weights(A, P.size(0))[0]  # [M]
        return P, W

    def _forward_with_attn(self, xb: torch.Tensor):
        """
        Runs forward_features once while capturing last-layer attention.
        Returns (out_dict, attn) where:
            out_dict: model.forward_features(xb)
            attn: [B, H, T, T] softmaxed attention or None if unavailable.
        """
        attn_maps = []
        last_attn = self.model.blocks[-1].attn

        def hook(module, inputs, output):
            x = inputs[0]                 # [B,T,C]
            B, T, C = x.shape
            # Case A: separate q_proj/k_proj (DINOv2)
            if hasattr(module, 'q_proj') and hasattr(module, 'k_proj'):
                q = module.q_proj(x); k = module.k_proj(x)
                H = module.num_heads; Dh = q.shape[-1] // H
                q = q.view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
                k = k.view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]
                attn = (q * (Dh ** -0.5)) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)              # [B,H,T,T]
            # Case B: fused qkv (timm-style)
            elif hasattr(module, 'qkv'):
                qkv = module.qkv(x)
                q, k, _ = qkv.chunk(3, dim=-1)
                H = module.num_heads; Dh = q.shape[-1] // H
                q = q.view(B, T, H, Dh).transpose(1, 2)
                k = k.view(B, T, H, Dh).transpose(1, 2)
                attn = (q * (Dh ** -0.5)) @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
            else:
                attn = None
            if attn is not None:
                attn_maps.append(attn.detach())

        h = last_attn.register_forward_hook(hook)
        out = self.model.forward_features(xb)
        h.remove()
        A = attn_maps[0] if len(attn_maps) else None
        return out, A

    def _cls_to_patch_weights(self, A: torch.Tensor, M: int) -> torch.Tensor:
        """
        A: [B,H,T,T] last-layer attention (softmaxed).
        M: number of patch tokens in out["x_norm_patchtokens"]
        Returns: [B,M] in [0,1], reduced over heads as configured.
        """
        B, Hh, T, _ = A.shape
        cls_to_all = A[:, :, 0, :]  # [B,H,T]
        if self.head_reduce == "mean":
            w_all = cls_to_all.mean(dim=1)   # [B,T]
        else:
            w_all = cls_to_all.max(dim=1).values

        # tokens layout: [CLS] [registers?] [patches]
        num_registers = max(0, T - 1 - M)
        start = 1 + num_registers
        w = w_all[:, start:start + M]        # [B,M]

        # normalize per image to [0,1]
        w = w - w.min(dim=1, keepdim=True).values
        denom = w.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        w = w / denom
        return w
