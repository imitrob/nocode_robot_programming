import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

import warnings
# silence the specific xFormers-not-available notices from DINOv2
warnings.filterwarnings(
    "ignore",
    message="xFormers is not available",
    category=UserWarning,
)

class DINOFeaturePresence:
    """ DINO patch embeddings.
    - Image score for class c = max_patch cosine(patch, proto_c).
    - Open-set gate: per-class threshold from training positives (percentile).
    
    Notation:
        N: Number of training samples
        C: Number of estimated classes
        H, W: Height, Weight of the image
        D: Embed dimension len (e.g., 384)
        B: Numbeer of samples in a batch
    """
    def __init__(
        self,
        dino_variant: str = "dinov2_vits14",
        input_size: int = 224,              # a bit larger than 224 to help small details
        percentile_keep: float = 0.10,      # 10th percentile acceptance (per-class)
        batch_size: int = 64,
    ):
        self.device = torch.device("cuda")
        self.model = torch.hub.load('facebookresearch/dinov2', dino_variant).to(self.device).eval()

        self.input_size = int(input_size)
        self.percentile_keep = float(percentile_keep)
        self.batch_size = int(batch_size)
        self.y_cls: List[str] = []  # class name list (index -> name)
        self.C: int = 0  # classes

        # MIL ATTENTION HEAD

        self.att_hidden = 128

        self.dropout_p = 0.2                 # TUNE
        self.dropout = torch.nn.Dropout(self.dropout_p)

        self.lr = 3e-4
        self.weight_decay = 5e-4
        self.mil_epochs = 120                # you can pass in
        self.warmup_epochs = 0               # small warm-up
        self.lambda_stable = 0.1             # weight for stable-region var penalty
        self.topk_frac = 0.10                # attention top-k pooling
        self.patience = 10                   # early-stopping patience







    def __str__(self):
        return self.__class__.__name__

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
        # 1. Compute patch features in batches

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
        N = X.shape[0] # samples
        C = len(y_cls)
        self.C: int = C
        y = y.long()
        self.y_cls = list(y_cls)

        # 1. Compute patch features in batches
        feats_list = []
        with torch.inference_mode():
            for i in range(0, N, self.batch_size):
                Xb = self._prep_batch(X[i:i+self.batch_size])
                with torch.amp.autocast(self.device.type):
                    Pb = self._patch_feats(Xb)          # [B, P, D]
                feats_list.append(Pb)
                # y_list.append(y[i:i+self.batch_size])
        patches = torch.cat(feats_list, dim=0)       # [N, P, D]

        N, M, D = patches.shape
        
        # Tiny MIL head
        self.att_hidden = 128
        self.W_att = torch.nn.Parameter(torch.randn(self.att_hidden, D, device=patches.device) * 0.02, requires_grad=True)
        self.b_att = torch.nn.Parameter(torch.zeros(self.att_hidden, device=patches.device), requires_grad=True)
        self.v_att = torch.nn.Parameter(torch.randn(self.att_hidden, device=patches.device) * 0.02, requires_grad=True)
        self.W_cls = torch.nn.Parameter(torch.randn(C, D, device=patches.device) * 0.02, requires_grad=True)
        self.dropout = torch.nn.Dropout(self.dropout_p)

        params = [self.W_att, self.b_att, self.v_att, self.W_cls]
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=self._lr_lambda)
        loss_fn = torch.nn.CrossEntropyLoss()


        train_idx, val_idx = self._split_train_val(N, y, val_frac=0.2)

        best_val, best_state, bad_epochs = -1.0, None, 0
        losses = []

        for ep in range(self.mil_epochs):        
    
            self.dropout.train()
            opt.zero_grad(set_to_none=True)

            P_tr = patches[train_idx]        # [Bt,M,D]
            y_tr = y[train_idx]              # [Bt]

            alpha = self._attention_weights(P_tr, self.W_att, self.b_att, self.v_att)      # [Bt,M]
            logits = self._class_logits(P_tr, self.W_cls, alpha, topk_frac=self.topk_frac) # [Bt,C]

            # Supervised loss (multi-class CE) ?
            L_sup = loss_fn(logits, y_tr)

            # 3) Stable-region variance penalty (add here)
            R_stable = self._stable_region_penalty(alpha, y_tr)
            loss = L_sup + self.lambda_stable * R_stable
            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()   # 2) cosine LR with warm-up

            # ---- VALIDATE ----
            self.dropout.eval()
            with torch.no_grad():
                P_va = patches[val_idx]
                y_va = y[val_idx]

                a_va = self._attention_weights(P_va, self.W_att, self.b_att, self.v_att)
                logits_va = self._class_logits(P_va, self.W_cls, a_va, topk_frac=self.topk_frac)
                pred = logits_va.argmax(dim=1)
                val_acc = (pred == y_va).float().mean().item()

            # early stopping on best val accuracy
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: p.detach().clone() for k,p in zip(["W_att","b_att","v_att","W_cls"], params)}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break  # stop early


        # load best head weights
        if best_state is not None:
            self.W_att.data.copy_(best_state["W_att"])
            self.b_att.data.copy_(best_state["b_att"])
            self.v_att.data.copy_(best_state["v_att"])
            self.W_cls.data.copy_(best_state["W_cls"])


        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3,3)); ax.plot(losses); ax.set(title="Loss", xlabel="epoch", ylabel="loss [-]")
        plt.tight_layout()

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, timestep: float | None = None) -> str:
        """
        MIL inference on one image.
        image: [H, W] float32 in [0,1] on CUDA (grayscale)
        returns: class name (str) from self.y_cls
        """
        # 1) DINO patch embeddings (frozen)
        Xb = self._prep_batch(image.unsqueeze(0))          # [1,3,S,S]
        with torch.amp.autocast(self.device.type):
            P = self._patch_feats(Xb)[0]                   # [M, D], unit-norm float32

        # 2) Normalize class weights for cosine-like scores
        Wcls = torch.nn.functional.normalize(self.W_cls, dim=1)   # [C, D]

        # 3) Attention scores: e_i = v^T tanh(W p_i + b)  →  α = softmax(e)
        #    (apply linear over last dim for all patches at once)
        Z = torch.tanh(torch.nn.functional.linear(P, self.W_att, self.b_att))  # [M, H]
        e = torch.nn.functional.linear(Z, self.v_att.unsqueeze(0), None).squeeze(-1)  # [M]
        alpha = torch.softmax(e, dim=0)                                        # [M]

        # 4) Class patch scores: s_{i,c} = w_c^T p_i   →   S_c via attended pooling
        #    Compute all s in one go: [M,D] @ [D,C] -> [M,C]
        s = P @ Wcls.T                                                         # [M, C]

        topk_frac = getattr(self, "topk_frac", 0.10)
        if topk_frac is None or topk_frac <= 0.0 or topk_frac >= 1.0:
            # Full attention sum: logits_c = Σ_i α_i * s_{i,c}
            logits = (alpha.unsqueeze(1) * s).sum(dim=0)                       # [C]
        else:
            # Top-k weighted mean (more stable than max)
            k = max(1, int(P.size(0) * topk_frac))
            vals, idx = torch.topk(alpha, k=k, largest=True, sorted=False)     # [k]
            s_topk = s.index_select(dim=0, index=idx)                           # [k, C]
            logits = (vals.unsqueeze(1) * s_topk).sum(dim=0) / (vals.sum() + 1e-8)  # [C]

        # 5) Argmax class -> name
        c_idx = int(torch.argmax(logits).item())
        
        name = self.y_cls[c_idx] if (self.y_cls and 0 <= c_idx < len(self.y_cls)) else str(c_idx)
        return name


    def predict_many(self, X: torch.Tensor) -> List[str]:
        return [self.predict(x) for x in X]# if X.ndim == 4 else X.unsqueeze(1))]
    
    def _attention_weights(self, P, W_att, b_att, v_att):
        """
        Args:
            P: [B,M,D] patch tokens (L2-normalized, float32, CUDA)
        Returns:
            alpha: [B,M]
            e_i = v^T tanh(W p_i + b);  alpha = softmax_i(e)
        """
        # linear over last dim: [B,M,D] -> [B,M,H]
        # Z_{n_,i} = tanh(W p_{n,i} + b)
        Z = torch.tanh(F.linear(P, W_att, b_att))          
        Z = self.dropout(Z)
        # e_{n,i} = v^T Z_{n,i}
        e = F.linear(Z, v_att.unsqueeze(0)).squeeze(-1) # [B, M]
        # softmax( e_{n,i} )
        alpha = torch.softmax(e, dim=1)  # over patches [B,M ]
        return alpha

    def _class_logits(self, P, W_cls, alpha, topk_frac=None):
        """
        Args:
            P: [B,M,D], W_cls: [C,D] (row-normalized), alpha: [B,M]
            s_{n,i,c} = w_c^T p_{n,i};  l_{n,c} = Σ_i alpha_{n,i} s_{n,i,c}  (or top-k weighted mean)
        Returns:
            logits: [B,C]
        """
        Wcls = F.normalize(W_cls, dim=1)  # cosine-like
        # Class patch scores
        s = torch.einsum('nmd,cd->nmc', P, Wcls)  # [B,M,C]

        # Attended pooling
        if topk_frac is None or not (0.0 < topk_frac < 1.0):
            logits = torch.einsum('nm,nmc->nc', alpha, s) # full attention sum
        else:
            k = max(1, int(P.size(1) * topk_frac))
            vals, idx = torch.topk(alpha, k=k, dim=1, largest=True, sorted=False)  # [B,k]
            s_topk = torch.gather(s, 1, idx.unsqueeze(-1).expand(-1,-1,s.size(2))) # [B,k,C]
            logits = (vals.unsqueeze(-1) * s_topk).sum(1) / (vals.sum(1, keepdim=True) + 1e-8)
        return logits

    def _stable_region_penalty(self, alpha, y):
        """ What changes in the trianing set is penalized, what stays the same is reinforced
            - This should cancel the background
            - per-class, per-patch variance of attention
            
        Stable-region variance penalty:
            R = Sum_c Sum_i Var_{n: y_n=c}( alpha_{n,i} )
            alpha: [B,M], y: [B]
        """
        R = alpha.new_zeros(())
        for c in torch.unique(y):
            mask = (y == c)
            if mask.sum() > 1:
                a_c = alpha[mask]                   # [N_c, M]
                # Var over images (dim=0) for each patch index i
        
                var_i = a_c.var(dim=0, unbiased=False)  # [M]
                R = R + var_i.sum()
        return R # stable

    def _split_train_val(self, N, y, val_frac=0.2):
        # y is 1D tensor on CUDA; move to CPU for splitting
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
        idx = torch.arange(N, device='cpu').numpy()
        y_cpu = y.detach().cpu().numpy()
        train_idx, val_idx = next(sss.split(idx, y_cpu))
        print("train_idx", train_idx, "val_idx", val_idx)
        return torch.tensor(train_idx, device=y.device), torch.tensor(val_idx, device=y.device)


    def _lr_lambda(self, epoch):
        # warm-up: linear from 0 -> 1 over warmup_epochs
        if epoch < self.warmup_epochs:
            return float(epoch + 1) / float(self.warmup_epochs)
        # cosine decay from 1 -> 0 over the remaining epochs
        t = (epoch - self.warmup_epochs) / max(1, (self.mil_epochs - self.warmup_epochs))
        return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * t))).item()
