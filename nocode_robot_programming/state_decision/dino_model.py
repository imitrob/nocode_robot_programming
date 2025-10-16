
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict
from typing import Tuple, Optional, List

class DINOStateDecider():
    """ Classify image to one of known `skill variations` (labels) using DINOv2 embeddings.
    If the best cosine similarity to any class centroid is below that class's
    learned threshold, return anomaly (-1).
    """
    def __init__(self,
                 dino_variant: str = "dinov2_vits14",
                 use_cls_token: bool = False,
                 batch_size: int = 64,
                 percent_keep: float = 0.05,    # keep rate for anomaly gate: 5th percentile of positives
                 ):
        """
        Args:
            dino_variant: one of {'dinov2_vits14','dinov2_vitb14','dinov2_vitl14',...}
            use_cls_token: if True use CLS token, else mean-pool patch tokens
            batch_size: batch size for training embeddings
            percent_keep: per-class acceptance percentile (0.05 = 5th percentile).
                          Stricter (lower) => more anomalies; looser (higher) => fewer anomalies.
            max_side: reserved for custom resizing, not crucial with 224 crops
        """
        self.y_cls = None

        self.device = torch.device("cuda") # we have only cuda machines

        # Note: torch.hub will download on first use; keep it outside hot loop
        self.model = torch.hub.load('facebookresearch/dinov2', dino_variant)
        self.model.eval().to(self.device)
        
        self.use_cls_token = use_cls_token
        self.batch_size = batch_size
        self.percent_keep = float(percent_keep)

        self.pre = transforms.Compose([
            transforms.ToPILImage() if hasattr(transforms, "ToPILImage") else (lambda x: Image.fromarray(x)),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # float32 [0,1]
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ])

        # learned state after train()
        self.emb_dim: Optional[int] = None
        self.class_centroids: Optional[torch.Tensor] = None  # [C, D], unit-norm
        self.class_labels: Optional[np.ndarray] = None       # [C] original label values
        self.class_thresholds: Optional[torch.Tensor] = None # [C] cosine threshold per class (float)
        self.train_embeddings: Optional[torch.Tensor] = None # [N, D] (kept if you want kNN fallback)
        self.train_labels: Optional[torch.Tensor] = None     # [N]


        self.pos_proto = None
        self.neg_proto = None


    @torch.inference_mode()
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        ''' Forward DINO '''
        # x: [B,3,224,224] float32/16 on device
        out = self.model.forward_features(x)
        if self.use_cls_token:
            feat = out["x_norm_clstoken"]          # [B, D]
        else:
            feat = out["x_norm_patchtokens"].mean(dim=1)  # [B, D]
        feat = F.normalize(feat, dim=-1)           # L2-normalize
        return feat

    def _prep_batch(self, imgs: List[np.ndarray]) -> torch.Tensor:
        # Convert list of HxW or HxWxC np arrays to a single tensor batch
        tensors = []
        for im in imgs:
            if im.ndim == 2:  # grayscale
                # ToPILImage in self.pre expects HxW or HxWxC uint8/float; ok
                pass
            elif im.ndim == 3 and im.shape[2] == 3:
                pass
            else:
                raise ValueError(f"Unsupported image shape {im.shape}")
            t = self.pre(im)  # [3,224,224], float32
            tensors.append(t)
        batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
        return batch
    
    def embed_batch(self, X: np.ndarray, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        X: (N, H, W) or (N, H, W, 3) uint8/float arrays.
        Returns: torch.Tensor (N, D) on CPU, L2-normalized.
        """
        if batch_size is None:
            batch_size = self.batch_size
        feats = []
        N = len(X)
        with torch.amp.autocast(self.device.type):
            for i in range(0, N, batch_size):
                batch_np = X[i:i+batch_size]
                xb = self._prep_batch(list(batch_np))
                fb = self._forward_features(xb)  # [B, D]
                feats.append(fb.detach().float().cpu())  # keep unified dtype on CPU
        return torch.cat(feats, dim=0)  # [N, D]


    def train(self, X: torch.Tensor, y: torch.Tensor, y_cls):
        """ X,y inputs are at cuda by default
        X: shape (N, H, W) or (N, H, W, 3) uint8/float
        y: shape (N,) integer/str labels (will be stored as original values)
        """
        X = X.cpu()
        y = y.cpu()
        self.y_cls = y_cls
        assert len(X) == len(y), "X and y must have the same length."

        # 1) Compute embeddings (batched, on device)
        feats = self.embed_batch(X)               # [N, D], CPU float32
        feats = F.normalize(feats, dim=-1)        # guard (already normalized but safe)
        self.emb_dim = feats.shape[1]
        y_np = np.asarray(y)
        y_unique = np.unique(y_np)

        # 2) Build class centroids
        centroids = []
        thresholds = []
        for cls in y_unique:
            idx = (y_np == cls).nonzero()[0]
            fcls = feats[idx]                    # [n_c, D]
            centroid = F.normalize(fcls.mean(dim=0, keepdim=True), dim=-1)  # [1, D]
            centroids.append(centroid)

            # 3) Per-class threshold from positives:
            # compute cosine to centroid for each training sample of that class
            sims = (fcls @ centroid.T).squeeze(1)  # [n_c]
            # choose the lower percentile (e.g., 5th) as accept threshold
            thr = np.percentile(sims.numpy(), self.percent_keep * 100.0)
            thresholds.append(float(thr))

        self.class_centroids = torch.cat(centroids, dim=0)     # [C, D]
        self.class_labels = y_unique.copy()
        self.class_thresholds = torch.tensor(thresholds, dtype=torch.float32)  # [C]

        # 4) (Optional) Keep all train embeddings for kNN fallback
        self.train_embeddings = feats
        self.train_labels = torch.from_numpy(y_np)

    @torch.inference_mode()
    def predict(self, image: np.ndarray, timestep: float | None = None) -> str:
        """ See state_decider.py:StateDeciderBase model
        """
        assert self.class_centroids is not None, "Call train() first."

        # 1) Embed single image
        xb = self._prep_batch([image])
        with torch.amp.autocast(self.device.type):
            feat = self._forward_features(xb)[0].float().cpu()   # [D]

        # 2) Cosine to all centroids
        centroids = self.class_centroids  # [C, D], CPU
        sims = (centroids @ feat)         # [C]
        best_idx = int(torch.argmax(sims).item())
        best_sim = float(sims[best_idx].item())
        best_label = self.class_labels[best_idx]

        # 3) Open-set gate: compare to that class's threshold
        if best_sim >= float(self.class_thresholds[best_idx]):
            # known
            # Return your original label type; ensure int if your system expects int ids
            try:
                lab_int = int(best_label)
            except Exception:
                # if labels were strings, you can map externally; here keep -1 fallback
                lab_int = int(best_idx)
            return self.y_cls[lab_int]
        else:
            return ""

   # -------- Optional: kNN fallback for edge cases --------
    @torch.inference_mode()
    def predict_knn(self, image: np.ndarray, k: int = 3) -> Tuple[bool, int]:
        """
        If you prefer instance-based prediction, use kNN in embedding space and the same open-set gate
        by comparing the sample to the centroid of the voted class.
        """
        assert self.train_embeddings is not None, "Train embeddings not available."
        xb = self._prep_batch([image])
        with torch.amp.autocast(self.device.type):
            feat = self._forward_features(xb)[0].float().cpu()   # [D]
        sims = (self.train_embeddings @ feat)                    # [N]
        topk = torch.topk(sims, k=min(k, sims.numel()), largest=True)
        neigh_idx = topk.indices
        neigh_labels = self.train_labels[neigh_idx].numpy()
        # majority vote
        vals, counts = np.unique(neigh_labels, return_counts=True)
        voted = vals[np.argmax(counts)]
        # gate by centroid of voted class
        cidx = int(np.where(self.class_labels == voted)[0][0])
        best_sim = float((self.class_centroids[cidx] @ feat).item())
        if best_sim >= float(self.class_thresholds[cidx]):
            try:
                return True, int(voted)
            except Exception:
                return True, int(cidx)
        return False, -1


    def predict_many(self, X):
        return [self.predict(x) for x in X]
