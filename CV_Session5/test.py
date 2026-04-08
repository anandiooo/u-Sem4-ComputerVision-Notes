import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./images/checker.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# harris -> array yg menanmpung angka-angka hasil deteksi corner
harris = cv2.cornerHarris(gray, 2, 5, 0.01)

harrisImg = img.copy()
harrisImg[harris > 0.01 * harris.max()] = [0, 0, 255]
harrisImg = cv2.cvtColor(harrisImg, cv2.COLOR_BGR2RGB)


# subpixel
_, threshold = cv2.threshold(harris, 0.01 * harris.max(), 255, cv2.THRESH_BINARY)
threshold = np.uint8(threshold)
_, _, _, cens = cv2.connectedComponentsWithStats(threshold)
cens = np.float32(cens)

corners = cv2.cornerSubPix(
    gray,
    cens,
    (5, 5),
    (-1, -1),
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
)

subpix = img.copy()
h, w = subpix.shape[:2]
for corner in corners:
    x, y = np.rint(corner).astype(int)
    if 0 <= x < w and 0 <= y < h:
        subpix[y, x] = [0, 0, 255]
subpix = cv2.cvtColor(subpix, cv2.COLOR_BGR2RGB)

images = {
	# 'Original': img,
	'Gray': gray,
	'Harris Corners': harrisImg,
	'Subpixel Corners': subpix
}

plt.figure(figsize=(15, 20))
for i, (title, image) in enumerate(images.items()):
	plt.subplot(3,1,i+1)
	plt.imshow(image, cmap='gray')
	plt.title(title)
	plt.axis('off')

plt.tight_layout()
plt.show()
