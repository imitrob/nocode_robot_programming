class SceneMatcher:
    def __init__(self, method="ORB", max_side=800, nfeatures=2000, ratio=0.75,
                 homography=True, ransac_thresh=3.0, maxIters=1000, confidence=0.99):
        self.method = method.upper()
        self.max_side = max_side
        self.ratio = ratio
        self.use_H = homography
        self.ransac_thresh = ransac_thresh
        self.maxIters = maxIters
        self.confidence = confidence

        if self.method == "SIFT":
            self.feat = cv2.SIFT_create(nfeatures=nfeatures)
            # FLANN KD-Tree for float descriptors
            index_params = dict(algorithm=1, trees=4)  # FLANN_INDEX_KDTREE = 1
            search_params = dict(checks=32)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            self.norm = cv2.NORM_L2
        elif self.method == "AKAZE":
            self.feat = cv2.AKAZE_create()  # descriptor is binary by default (MLDB)
            # FLANN LSH for binary descriptors
            index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                                table_number=12, key_size=20, multi_probe_level=2)
            search_params = dict(checks=64)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            self.norm = cv2.NORM_HAMMING
        else:  # ORB
            self.feat = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=15, scaleFactor=1.2, nlevels=8)
            index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                                table_number=12, key_size=20, multi_probe_level=2)
            search_params = dict(checks=64)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            self.norm = cv2.NORM_HAMMING

        # Pick the best available robust method
        self._usac = getattr(cv2, "USAC_FAST", None)
        self._magsac = getattr(cv2, "USAC_MAGSAC", None)

    def _prep(self, img):
        # to grayscale + downscale, preserving aspect ratio
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        s = self.max_side / max(h, w)
        if s < 1.0:
            img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        return img

    def similar_scene(self, img1, img2, min_good=8):
        img1 = self._prep(img1)
        img2 = self._prep(img2)

        k1, d1 = self.feat.detectAndCompute(img1, None)
        k2, d2 = self.feat.detectAndCompute(img2, None)
        if d1 is None or d2 is None or len(k1) < min_good or len(k2) < min_good:
            return 0.0, None

        # FLANN + KNN
        knn = self.matcher.knnMatch(d1, d2, k=2)
        # Lowe ratio
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)

        if len(good) < min_good:
            return 0.0, None

        pts1 = np.float32([k1[m.queryIdx].pt for m in good])
        pts2 = np.float32([k2[m.trainIdx].pt for m in good])

        # Robust geometry
        mask = None
        if self.use_H:
            method = cv2.RANSAC
            if self._usac is not None:   # prefer USAC if available
                method = self._usac
            elif self._magsac is not None:
                method = self._magsac

            H, inliers = cv2.findHomography(
                pts1, pts2, method,
                ransacReprojThreshold=self.ransac_thresh,
                maxIters=self.maxIters, confidence=self.confidence
            )
            mask = inliers
            M = H
        else:
            method = getattr(cv2, "USAC_FAST", cv2.RANSAC)
            F, inliers = cv2.findFundamentalMat(
                pts1, pts2, method, ransacReprojThreshold=self.ransac_thresh,
                confidence=self.confidence, maxIters=self.maxIters
            )
            mask = inliers
            M = F

        inlier_ratio = float(mask.sum()) / len(good) if mask is not None else 0.0
        return inlier_ratio, M
sm = SceneMatcher()