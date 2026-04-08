import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os

PATH = 'images/target/'
object = cv2.imread(PATH + 'hina.png')
object = cv2.cvtColor(object, cv2.COLOR_BGR2RGB)

DATA_PATH = 'images/source/'
data = []

best_matches = 0

for image_path in os.listdir(DATA_PATH):
    image_path = DATA_PATH + image_path
    image_data = cv2.imread(image_path)
    data.append(image_data)
    
grayscale_object = cv2.cvtColor(object, cv2.COLOR_RGB2GRAY)
grayscale_object = cv2.GaussianBlur(grayscale_object, (3,3), 0)

# SIFT - Scale Invariant Feature Transform
sift = cv2.SIFT_create()
# ORB - Oriented FAST and Rotated Brief
orb = cv2.ORB_create()
# AKAZE - Accelerated KAZE
akaze = cv2.AKAZE_create()

target_keypoint, target_descriptor = sift.detectAndCompute(grayscale_object, None)
target_descriptor = np.float32(target_descriptor)

for index, img in enumerate(data):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grayscale_data = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grayscale_data = cv2.GaussianBlur(grayscale_data, (3,3), 0)
    img_keypoint, img_descriptor = sift.detectAndCompute(grayscale_data, None)
    img_descriptor = np.float32(img_descriptor)

    # feature matching
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    match = flann.knnMatch(target_descriptor, img_descriptor, 2)

    # match masking
    matchesmask = [[0,0]for _ in range (0,len(match))]
    current_match = 0

    # Lowe's ratio
    for i, (fm,sm) in enumerate(match):
        if fm.distance < 0.7 * sm.distance:
            matchesmask[i] = [1,0]
            current_match+=1

    score = 0
    if current_match > 0:
        valid_distances = [m.distance for i, (m, n) in enumerate(match) if matchesmask[i][0] == 1]
        score = sum(valid_distances) / len(valid_distances) if valid_distances else 0

    if best_matches<current_match:
        best_matches = current_match
        best_matches_data = {
            'image_data' : img,
            'keypoint' : img_keypoint,
            'descriptor' : img_descriptor,
            'match' : match,
            'matchesmask': matchesmask,
            'score' : score 
        }

# plotting
result = cv2.drawMatchesKnn(
    object,
    target_keypoint,
    best_matches_data['image_data'],
    best_matches_data['keypoint'],
    best_matches_data['match'],
    None,
    matchesMask=best_matches_data['matchesmask'],
    matchColor=[20,150,155],
    # singlePointColor=[0,0,255]
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
)
  

plt.figure()
plt.imshow(result)
plt.title(f"Best Match Result: {best_matches_data['score']}")
plt.axis(False)
plt.show()
    