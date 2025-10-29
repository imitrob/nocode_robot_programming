
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from nocode_robot_programming.state_decision.dino_model import DINOFeaturePresence

class DINOWithMIL(DINOFeaturePresence):
    """ Multiple-Instance Learning (MIL) over DINO patch tokens. 
    
    H: int Hidden params
    D: int Number of features (embeddings) 
    C: int Number of classes for classification
    W: int width pixels
    H: int height pixels
    N: int Number of training samples
    """
    def __init__(
        self,
        *args,
        att_hidden: int = 128,
        dropout_p: float = 0.1,
        lr: float = 7e-5,
        weight_decay: float = 1e-3,
        epochs: int = 1000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.H = int(att_hidden)
        self.dropout_p = float(dropout_p)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)

        # will be created in train()
        self.W_att: Optional[torch.Tensor] = None  # [H, D]
        self.b_att: Optional[torch.Tensor] = None  # [H]
        self.v_att: Optional[torch.Tensor] = None  # [H]
        self.W_cls: Optional[torch.Tensor] = None  # [C, D]

        self._dropout = nn.Dropout(self.dropout_p)

    def __str__(self):
        return str(self.__class__.__name__)

    def train(self, X: torch.Tensor, y: torch.Tensor, y_cls: List[str]) -> None:
        """ Supervised MIL training of attention head + cosine classifier

        X: [N, H, W] float32 in [0,1]
        y: [N] int64 (class indices)
        """
        assert X.ndim == 3, "X must be [N,H,W]"
        self.y_cls = list(y_cls)
        C = len(y_cls)
        y = y.to(self.device).long()

        # 1) Extract all patch features (frozen DINO)
        P = self._all_patch_feats(X)  # [N, M, D] created under inference_mode
        P = P.detach().clone()
        N, M, D = P.shape

        # 2) Initialize MIL head + class weights (cosine)
        self._init_head(D, C)

        params = [self.W_att, self.b_att, self.v_att, self.W_cls]
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        # 3) Simple full-batch training (no validation)
        for _ in range(self.epochs):
            self._dropout.train()
            opt.zero_grad(set_to_none=True)

            alpha = self._attention_weights(P)  # [N, M]
            logits = self._class_logits(P, alpha)  # [N, C]
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

        # 4) Optional open-set thresholds based on train scores
        if self.percentile_keep is not None:
            with torch.inference_mode():
                a_all = self._attention_weights(P)
                logits_all = self._class_logits(P, a_all)  # [N, C]
            thresholds = []
            for c in range(C):
                s = logits_all[y == c, c].detach().float().cpu().sort().values
                if len(s) == 0:
                    thresholds.append(float("-inf"))
                else:
                    k = max(0, min(len(s)-1, int(round(self.percentile_keep * (len(s)-1)))))
                    thresholds.append(s[k].item())
            self.thresholds = torch.tensor(thresholds, device=self.device)

    @torch.inference_mode()
    def predict(self, image: torch.Tensor) -> str:
        """
        image: [H, W] float32 in [0,1]
        """
        assert self.W_att is not None and self.W_cls is not None, "Call train() first"
        p = self._single_patch_feats(image)  # [M, D]
        a = self._attention_weights(p.unsqueeze(0))[0]  # [M]
        logits = self._class_logits(p.unsqueeze(0), a.unsqueeze(0))[0]  # [C]
        c = int(torch.argmax(logits).item())
        if self.percentile_keep is not None and self.thresholds is not None:
            if logits[c] < self.thresholds[c]:
                return "unknown"
        return self.y_cls[c]

    def _pool_patches(self, P: torch.Tensor) -> torch.Tensor:
        """ Fallback pooling before training initializes MIL params """
        if self.W_att is None:
            G = P.mean(dim=1)
            return F.normalize(G, dim=-1)
        alpha = self._attention_weights(P)  # [N, M]
        G = torch.einsum('nm,nmd->nd', alpha, P)  # [N, D]
        return F.normalize(G, dim=-1)

    def _init_head(self, D: int, C: int) -> None:
        device = self.device
        W_att = torch.empty(self.H, D, device=device)
        torch.nn.init.xavier_uniform_(W_att)
        b_att = torch.zeros(self.H, device=device)
        v_att = 0.01 * torch.randn(self.H, device=device)

        W_cls = torch.randn(C, D, device=device)
        W_cls = F.normalize(W_cls, dim=1)

        self.W_att = nn.Parameter(W_att, requires_grad=True)
        self.b_att = nn.Parameter(b_att, requires_grad=True)
        self.v_att = nn.Parameter(v_att, requires_grad=True)
        self.W_cls = nn.Parameter(W_cls, requires_grad=True)

    def _attention_weights(self, P: torch.Tensor) -> torch.Tensor:
        """
        P: [N, M, D] -> alpha: [N, M] with sum(alpha)=1 per sample.
        e_i = v^T tanh(W p_i + b); alpha = softmax(e)
        """
        Z = torch.tanh(F.linear(P, self.W_att, self.b_att))  # [N, M, H]
        Z = self._dropout(Z)
        e = F.linear(Z, self.v_att.unsqueeze(0)).squeeze(-1)  # [N, M]
        alpha = torch.softmax(e, dim=1)
        return alpha

    def _class_logits(self, P: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """ Cosine logits from attention-weighted pooled features
        P: [N, M, D], alpha: [N, M] -> [N, C]
        """
        G = torch.einsum('nm,nmd->nd', alpha, P)  # [N, D]
        G = F.normalize(G, dim=1)
        W = F.normalize(self.W_cls, dim=1)  # [C, D]
        return torch.einsum('nd,cd->nc', G, W)
