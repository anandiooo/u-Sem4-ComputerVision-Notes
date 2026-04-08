import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#setup
target_path = './images/target/'
data_path = './images/source/'

#prepro
object = cv2.imread(target_path + 'hina.png')
object = cv2.cvtColor(object, cv2.COLOR_BGR2RGB)
gray_scale = cv2.cvtColor(object, cv2.COLOR_RGB2GRAY)
gray_scale = cv2.GaussianBlur(gray_scale, (3,3), 0)

data = []
for image_path in os.listdir(data_path):
    image_data = cv2.imread(data_path + image_path)
    data.append(image_data)

#detect
akaze = cv2.AKAZE_create()
best_matches = 0

target_keypoint, target_descriptor = akaze.detectAndCompute(gray_scale, None)
target_descriptor = np.float32(target_descriptor)

#feature matching
for idx, img in enumerate(data):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_scale_data = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_scale_data = cv2.GaussianBlur(gray_scale_data, (3,3), 0)

    img_keypoint, img_descriptor = akaze.detectAndCompute(gray_scale_data, None)
    img_descriptor = np.float32(img_descriptor)

    flann = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), dict(checks = 50))
    match = flann.knnMatch(target_descriptor, img_descriptor, 2)

    matchesMask = [[0,0] for _ in range (0,len(match))]
    current_match = 0

    for i, (fm, sm) in enumerate(match):
        if fm.distance < 0.7 * sm.distance:
            matchesMask[i] = [1,0]
            current_match+=1

    if best_matches < current_match:
        best_matches = current_match
        best_matches_data = {
            'image_data' : img,
            'keypoint' : img_keypoint,
            'descriptor' : img_descriptor,
            'match' : match,
            'matchesmask' : matchesMask
        }

#inference
result = cv2.drawMatchesKnn(
    object,
    target_keypoint,
    best_matches_data['image_data'],
    best_matches_data['keypoint'],
    best_matches_data['match'],
    outImg=None,
    matchesMask = best_matches_data['matchesmask'],
    matchColor=(0,0,255),
    flags= cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
)

plt.figure()
plt.imshow(result)
plt.show()
