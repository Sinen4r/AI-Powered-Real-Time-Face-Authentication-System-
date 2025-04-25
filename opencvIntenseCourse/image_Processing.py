import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import pprint


'''Hue: The color type (0-179 in OpenCV, where 0=Red, 60=Green, 120=Blue).

Saturation: Intensity of the color (0=gray, 255=full color).

Value: Brightness (0=black, 255=bright).'''
# Changing Colorspaces


# cap = cv.VideoCapture(0)

# while True:
#     _, img = cap.read()
#     hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#     lower_blue = np.array([100, 150, 0])
#     upper_blue = np.array([140, 255, 255])
#     mask_blue = cv.inRange(hsv, lower_blue, upper_blue)

#     lower_green = np.array([40, 50, 50])
#     upper_green = np.array([80, 255, 255])
#     mask_green = cv.inRange(hsv, lower_green, upper_green)

#     lower_red1 = np.array([0, 120, 70])
#     upper_red1 = np.array([10, 255, 255])
#     mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)

#     lower_red2 = np.array([170, 120, 70])
#     upper_red2 = np.array([180, 255, 255])
#     mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
#     mask_red = cv.bitwise_or(mask_red1, mask_red2)
#     combined_mask = cv.bitwise_or(cv.bitwise_or(mask_blue, mask_green), mask_red)
#     res = cv.bitwise_and(img, img, mask=combined_mask)
#     cv.imshow("frame", img)
#     cv.imshow("mask", combined_mask)
#     cv.imshow("result", res)
#     k = cv.waitKey(5) & 0xFF
#     if k == 27:
#         break
# cap.release()
# cv.destroyAllWindows()



#Geometric Transformations of Images

# cap=cv.VideoCapture(0)
# while True:
#     _,img=cap.read()
#     res=cv.resize(img,None,fx=2,fy=2,interpolation=cv.INTER_NEAREST)
#     cv.imshow("res",res)
#     k = cv.waitKey(5) & 0xFF
#     if k == 27:
#          break

# img=cv.imread("logoo.jpg",cv.IMREAD_GRAYSCALE)
# rows,coloumns=img.shape
# M = np.float32([[1,0,100],[0,1,50]])
# dst=cv.warpAffine(img,M,(coloumns,rows)) #dst(width, height)
# cv.imshow("s",dst)
# cv.waitKey(0)

def nothing(x):
    pass
img=cv.imread("logoo.jpg",cv.IMREAD_GRAYSCALE)
rows,columns=img.shape
# cv.namedWindow('image')

# cv.createTrackbar('B','image',0,360,nothing)

# while True:
#     k = cv.waitKey(1) & 0xFF
#     if k == 27:
#         break
#     r= cv.getTrackbarPos('B','image')
#     M=cv.getRotationMatrix2D(((columns-1)/2,(rows-1)/2),r,1)
#     dst = cv.warpAffine(img,M,(columns,rows))
#     cv.imshow("image",dst)

 
# pts1 = np.float32([[50,50],[200,50],[50,200]])
# pts2 = np.float32([[10,100],[200,50],[100,250]])

# M = cv.getAffineTransform(pts1,pts2)
 
# dst = cv.warpAffine(img,M,(columns,rows))
 
# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()

#Perspective
# pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
# pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
# print(pts1)
# M=cv.getPerspectiveTransform(pts1,pts2)
# dst = cv.warpPerspective(img,M,(300,300))

# plt.subplot(121),plt.imshow(img),plt.title("original")
# plt.subplot(121),plt.scatter([ i[0] for i in pts1],[i[1] for i in pts1],s=50, c='red')
# plt.subplot(122),plt.imshow(dst),plt.title("output")
# plt.show()
import math

# img=cv.imread("gradient.PNG")
# img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('fs',img)
# cv.waitKey(0)
# print(img)
# _,thresh1=cv.threshold(img,127,255,cv.THRESH_BINARY)
# _,thresh2=cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
# _,thresh3=cv.threshold(img,127,255,cv.THRESH_TRUNC)
# _,thresh4=cv.threshold(img,127,255,cv.THRESH_TOZERO)
# _,thresh5=cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)

# titles=["original","binray","binray inverse","trunc","tozero","tozero inverse"]
# images=[img,thresh1,thresh2,thresh3,thresh4,thresh5]
# for i in range (6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray'),plt.title(titles[i])

# plt.show()
img = cv.imread('logoo.jpg', cv.IMREAD_GRAYSCALE)
img = cv.medianBlur(img,5)
ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

 
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
 
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()