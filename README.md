# Computer Vision Quiz Summary (6 Sessions)

This note is made for fast review before quiz. Focus on workflow + function names.

## Super Short Memory Flow (Follow This Order)

**List -> Load -> Smooth -> Detect -> Match -> Filter -> Draw -> Show**

If you forget everything, remember this order.

## Grading Rubric Cheat Sheet

### 1) Preprocess (30)

#### A. Listing image (10)

- Goal: collect candidate images from folder.
- Main functions:
  - `os.listdir(DATA_PATH)`
  - `os.path.join(DATA_PATH, filename)`
  - `os.path.isfile(path)`
  - `filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))`

Example pattern:

```python
for filename in os.listdir(DATA_PATH):
    path = os.path.join(DATA_PATH, filename)
    if not os.path.isfile(path):
        continue
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        continue
```

#### B. Load image (10)

- Goal: read image correctly, then convert color space if needed.
- Main functions:
  - `cv2.imread(path)`
  - `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`
  - `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
  - `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`

#### C. Smoothing image (10)

- Goal: reduce noise before thresholding or feature extraction.
- Main functions:
  - `cv2.blur(img, (7, 7))` (mean blur)
  - `cv2.medianBlur(img, 5 or 7)`
  - `cv2.GaussianBlur(img, (7, 7), sigma)`
  - `cv2.bilateralFilter(img, 9, 75, 75)`
- In your matching pipeline, `medianBlur` is used before feature detection.

### 2) Feature Detect (15)

- Goal: find keypoints and descriptors.
- Main functions in your sessions:
  - `cv2.cornerHarris(...)` (corner response)
  - `cv2.cornerSubPix(...)` (refine corner location)
  - `cv2.FastFeatureDetector_create()`
  - `cv2.ORB_create()`
  - `cv2.SIFT_create(...)`
  - `detector.detect(gray, None)`
  - `sift.detectAndCompute(gray, None)`

### 3) Feature Matching (35)

#### A. Match (15)

- Goal: compare descriptors between target and candidate image.
- Main functions:
  - `cv2.FlannBasedMatcher(indexParams, searchParams)`
  - `flann.knnMatch(targetDesc, imgDesc, k=2)`

#### B. Filter match (20)

- Goal: keep only reliable matches.
- Method used in your notebook: **Lowe's ratio test**.

Rule:

```python
if fm.distance < 0.75 * sm.distance:
    # good match
```

- Main pattern:
  - Create `matchMask = [[0, 0] for _ in range(len(matches))]`
  - For each pair `(fm, sm)`, apply ratio test
  - Mark good match mask and count good matches
  - Keep image with highest good match count as best result

### 4) Result (20)

#### A. Draw match (10)

- Goal: visualize correspondence lines/points.
- Main functions:
  - `cv2.drawMatchesKnn(...)`
  - `flags=cv2.DrawMatchesFlags_DEFAULT`
  - `matchesMask=bestResult["matchMask"]`

#### B. Show match (10)

- Goal: display final output clearly.
- Main functions:
  - `plt.figure(figsize=(...))`
  - `plt.imshow(resultImg)`
  - `plt.title(...)`
  - `plt.axis("off")`
  - `plt.show()`

---

## Folder-by-Folder Summary (What Each Session Taught)

### 1) CV_Session1_Python_Numpy_OpenCV

- Python review: input/output, if-else, loop, function, data structure.
- NumPy basics: array, shape, zeros/ones/empty, stack.
- OpenCV basics: image load/show and simple color-channel manipulation.
- Core functions: `cv.imread`, `cv.resize`, `cv.imshow`, `cv.waitKey`.

### 2) CV_Session2

- Basic image pipeline: load -> resize -> grayscale.
- Histogram understanding + manual intensity counting.
- Histogram equalization for contrast improvement.
- Core functions: `cv2.imread`, `cv2.resize`, `cv2.cvtColor`, `cv2.equalizeHist`.

### 3) CV_Session3_Threshold_dll_lupa

- Thresholding types: binary, inverse, trunc, tozero, otsu.
- Adaptive thresholding: mean and gaussian.
- Filtering/smoothing comparison: mean, median, gaussian, bilateral.
- Core functions: `cv2.threshold`, `cv2.adaptiveThreshold`, `cv2.blur`, `cv2.medianBlur`, `cv2.GaussianBlur`, `cv2.bilateralFilter`.

### 4) CV_Session4

- Edge detection concepts and implementation.
- Laplacian (2nd derivative), Sobel X/Y (1st derivative), gradient magnitude, Canny.
- Core functions: `cv2.Laplacian`, `cv2.Sobel`, `cv2.Canny`.

### 5) CV_Session5

- Feature detection deep dive.
- Harris corner, subpixel corner refinement, FAST and ORB keypoints.
- Image-folder loop pattern for running detector across many images.
- Core functions: `cv2.cornerHarris`, `cv2.cornerSubPix`, `cv2.FastFeatureDetector_create`, `cv2.ORB_create`, `cv2.drawKeypoints`.

### 6) CV_Session6

- Full matching pipeline (closest to rubric):
  - list candidate images
  - preprocess target and candidates (gray + smoothing + CLAHE)
  - detect features with SIFT
  - match with FLANN + KNN
  - filter with ratio test
  - keep best match and draw result
- Core functions: `cv2.createCLAHE`, `cv2.SIFT_create`, `detectAndCompute`, `cv2.FlannBasedMatcher`, `knnMatch`, `cv2.drawMatchesKnn`.

---

## 30-Second Last-Minute Recall

1. **Preprocess**: `os.listdir` -> `cv2.imread` -> `cvtColor` -> `medianBlur/GaussianBlur`.
2. **Detect**: `SIFT_create` + `detectAndCompute` (or ORB/FAST/Harris).
3. **Match**: `FlannBasedMatcher` + `knnMatch(k=2)`.
4. **Filter**: ratio test `m.distance < 0.75*n.distance`.
5. **Result**: `drawMatchesKnn` -> `plt.imshow` -> `plt.show`.

Good luck for your quiz.
