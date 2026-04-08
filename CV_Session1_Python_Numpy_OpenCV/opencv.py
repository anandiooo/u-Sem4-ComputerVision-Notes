# openCV
import cv2 as cv

# masukin image ke code
img = cv.imread("./assets/art.jpg")
# print(img.shape) # 1200,1200,3
# img = cv.resize(img, (256,256))
# cv.imshow("tes", img)
# cv.waitKey(0)

modifiedImage = img.copy()
modifiedImage = cv.resize(modifiedImage, (256,256))
modifiedImage[:,:128,0] = 0
modifiedImage[:,:128,1] = 0
modifiedImage[:,:128,2] = 0
# cv2 ngeload warna (B,G,R)
cv.imshow("tes", modifiedImage)
cv.waitKey(0)