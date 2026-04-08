import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# CV-SPDFI (quick memory)
# S = Setup            -> import + path
# P = Preprocess       -> list image + load image + smoothing
# D = Detect           -> keypoints/descriptors
# F = Find+Filter      -> match + ratio test
# I = Inspect          -> draw match + show match

# [S] Setup
target_path = './CV_Session6/Dataset/Object.jpg'
data_path = './CV_Session6/Dataset/Data/'


# [P] Preprocess (list + load + smoothing)
target_bgr = cv2.imread(target_path)
target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2GRAY)
smooth_gray = cv2.medianBlur(target_gray, (3,3), 0)

data = []
for image_name in os.listdir(data_path):
    image_bgr = cv2.imread(os.path.join(data_path, image_name))
    if image_bgr is None:
        continue
    data.append((image_name, image_bgr, smooth_gray(image_bgr)))


# [D] Detect
akaze = cv2.AKAZE_create()
target_keypoint, target_descriptor = akaze.detectAndCompute(target_gray, None)
if target_descriptor is None or len(target_keypoint) < 2:
    raise RuntimeError('Target feature detection failed.')
target_descriptor = np.float32(target_descriptor)


# [F] Find+Filter (match + ratio test)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
best_matches = 0
best_matches_data = None

for image_name, image_bgr, image_gray in data:
    img_keypoint, img_descriptor = akaze.detectAndCompute(image_gray, None)
    if img_descriptor is None or len(img_keypoint) < 2:
        continue

    img_descriptor = np.float32(img_descriptor)
    match = flann.knnMatch(target_descriptor, img_descriptor, 2)

    matchesMask = [[0, 0] for _ in range(len(match))]
    current_match = 0

    for i, pair in enumerate(match):
        if len(pair) < 2:
            continue
        fm, sm = pair
        if fm.distance < 0.7 * sm.distance:
            matchesMask[i] = [1, 0]
            current_match += 1

    if best_matches < current_match:
        best_matches = current_match
        best_matches_data = {
            'name': image_name,
            'image_data': cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            'keypoint': img_keypoint,
            'descriptor': img_descriptor,
            'match': match,
            'matchesmask': matchesMask,
        }


# [I] Inspect (draw + show)
if best_matches_data is None:
    raise RuntimeError('No good matches found.')

result = cv2.drawMatchesKnn(
    target_rgb,
    target_keypoint,
    best_matches_data['image_data'],
    best_matches_data['keypoint'],
    best_matches_data['match'],
    outImg=None,
    matchesMask=best_matches_data['matchesmask'],
    matchColor=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
)

plt.figure()
plt.imshow(result)
plt.title(f"Best: {best_matches_data['name']} | Good Matches: {best_matches}")
plt.axis('off')
plt.show()
