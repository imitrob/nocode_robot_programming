import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import torch

class StateDeciderSIFT:  # Fits StateDeciderBase interface
    def __init__(self,
                 method: str = "SIFT",
                 max_side: int = 800,
                 nfeatures: int = 2000,
                 ratio: float = 0.75,
                 homography: bool = True,
                 ransac_thresh: float = 3.0,
                 maxIters: int = 1000,
                 confidence: float = 0.99,
                 min_good: int = 8,
                 max_refs_per_class: int = 5,
                 anomaly_percentile: float = 0.10   # 10th percentile as acceptance cutoff
                 ):
        """
        method: 'SIFT' | 'AKAZE' | 'ORB'
        anomaly_percentile: lower -> stricter (more anomalies), higher -> looser
        """
        self.y_cls = None
        self.method = method.upper()
        self.max_side = max_side
        self.ratio = ratio
        self.use_H = homography
        self.ransac_thresh = ransac_thresh
        self.maxIters = maxIters
        self.confidence = confidence
        self.min_good = min_good
        self.max_refs_per_class = max_refs_per_class
        self.anomaly_percentile = anomaly_percentile

        # --- feature + matcher setup (reused) ---
        if self.method == "SIFT":
            self.feat = cv2.SIFT_create(nfeatures=nfeatures)
            index_params = dict(algorithm=1, trees=4)  # KD-Tree
            search_params = dict(checks=32)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            self.norm = cv2.NORM_L2
        elif self.method == "AKAZE":
            self.feat = cv2.AKAZE_create()  # binary MLDB
            index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)  # LSH
            search_params = dict(checks=64)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            self.norm = cv2.NORM_HAMMING
        else:  # ORB
            self.feat = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=15, scaleFactor=1.2, nlevels=8)
            index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)  # LSH
            search_params = dict(checks=64)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            self.norm = cv2.NORM_HAMMING

        # Prefer USAC if available
        self._usac = getattr(cv2, "USAC_FAST", None)
        self._magsac = getattr(cv2, "USAC_MAGSAC", None)

        # learned after train()
        self.refs_by_class: Dict[Any, List[Dict[str, Any]]] = {}
        self.threshold_by_class: Dict[Any, float] = {}

    # ------------- utils -------------
    def _prep(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        s = self.max_side / max(h, w)
        if s < 1.0:
            img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        return img

    def _detect(self, img: np.ndarray):
        img = img.squeeze(0).detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        k, d = self.feat.detectAndCompute(img, None)
        return k, d

    def _score_pair(self, k1, d1, k2, d2) -> Tuple[float, Optional[np.ndarray]]:
        if d1 is None or d2 is None or len(k1) < self.min_good or len(k2) < self.min_good:
            return 0.0, None

        knn = self.matcher.knnMatch(d1, d2, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2: continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)
        if len(good) < self.min_good:
            return 0.0, None

        pts1 = np.float32([k1[m.queryIdx].pt for m in good])
        pts2 = np.float32([k2[m.trainIdx].pt for m in good])

        if self.use_H:
            method = cv2.RANSAC
            if self._usac is not None:
                method = self._usac
            elif self._magsac is not None:
                method = self._magsac
            H, inliers = cv2.findHomography(pts1, pts2, method,
                                            ransacReprojThreshold=self.ransac_thresh,
                                            maxIters=self.maxIters, confidence=self.confidence)
            mask = inliers
        else:
            method = getattr(cv2, "USAC_FAST", cv2.RANSAC)
            F, inliers = cv2.findFundamentalMat(pts1, pts2, method,
                                                ransacReprojThreshold=self.ransac_thresh,
                                                confidence=self.confidence, maxIters=self.maxIters)
            mask = inliers

        inlier_ratio = float(mask.sum()) / len(good) if mask is not None else 0.0
        return inlier_ratio, mask

    # ------------- training -------------
    def train(self, X: np.ndarray, y: np.ndarray, y_cls):
        """
        X: (N, H, W) or (N, H, W, 3)  uint8/float
        y: (N,) labels (ints/strings)
        """
        self.y_cls = y_cls
        assert len(X) == len(y)
        X = [self._prep(x) for x in X]

        # extract features for all images once
        feats = [self._detect(x) for x in X]

        # group by class
        by_cls: Dict[Any, List[int]] = defaultdict(list)
        for i, cls in enumerate(y):
            by_cls[cls].append(i)

        self.refs_by_class = {}
        self.threshold_by_class = {}

        for cls, idxs in by_cls.items():
            # pick up to K reference exemplars (evenly spaced)
            K = min(self.max_refs_per_class, len(idxs))
            if K <= 0: 
                continue
            # evenly spaced selection
            if K < len(idxs):
                step = max(1, len(idxs) // K)
                chosen = idxs[::step][:K]
            else:
                chosen = idxs

            refs = []
            for i in chosen:
                k, d = feats[i]
                refs.append({"kp": k, "desc": d})

            # compute positive scores (image vs refs) to set threshold
            pos_scores = []
            for i in idxs:
                kq, dq = feats[i]
                if dq is None or len(kq) < self.min_good:
                    continue
                # best score against refs of the same class
                best = 0.0
                for r in refs:
                    s, _ = self._score_pair(kq, dq, r["kp"], r["desc"])
                    if s > best: best = s
                if best > 0:
                    pos_scores.append(best)

            # fallback threshold if not enough matches
            if len(pos_scores) >= 5:
                thr = float(np.percentile(pos_scores, self.anomaly_percentile * 100.0))
            elif len(pos_scores) > 0:
                thr = float(min(pos_scores)) * 0.9  # conservative
            else:
                thr = 0.25  # reasonable default; tune if needed

            self.refs_by_class[cls] = refs
            self.threshold_by_class[cls] = thr

    # ------------- inference -------------
    def predict(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Returns: (is_known, label_or_-1)
        """
        assert len(self.refs_by_class) > 0, "Call train() first."
        img = self._prep(image)
        kq, dq = self._detect(img)
        if dq is None or len(kq) < self.min_good:
            return (False, "")

        best_cls = None
        best_score = 0.0

        # score against class refs (take max per class)
        for cls, refs in self.refs_by_class.items():
            cls_best = 0.0
            for r in refs:
                s, _ = self._score_pair(kq, dq, r["kp"], r["desc"])
                if s > cls_best:
                    cls_best = s
            if cls_best > best_score:
                best_score = cls_best
                best_cls = cls

        # anomaly gate
        thr = self.threshold_by_class.get(best_cls, 0.25)
        if best_score >= thr:
            ret = best_cls if not torch.is_floating_point(best_cls) else int(list(self.refs_by_class.keys()).index(best_cls))
            return True, self.y_cls[ret]
        else:
            return (False, "")

    def __call__(self, image: np.ndarray, timestep: float) -> Tuple[bool, int]:
        known, lab = self.predict(image)
        return (known, lab if isinstance(lab, int) else (-1 if not known else int(lab)))
