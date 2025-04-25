import cv2 as cv
import numpy as np 
import sys

img=cv.imread("tata.png")
assert img is  None,"file could not be read"

#Image loaded by default in bgr format

# pixel=img[200,200]
# #[b,g,r]
# print(img.size)
# #costly operation split ->better use numpy splitting
# b,g,r=cv.split(img)
# # img=cv.merge((r,g,b))
# # img=cv.merge((b,r,g))
# img[:,:,0]=0
# cv.imwrite("rgb.png",img)

from matplotlib import pyplot as plt

# BLUE=[255,0,0]

# img=cv.imread("logoo.jpg")
# replicate=cv.copyMakeBorder(img,20,20,20,20,cv.BORDER_REPLICATE)
# reflect=cv.copyMakeBorder(img,20,20,20,20,cv.BORDER_REFLECT)
# reflect_201=cv.copyMakeBorder(img,20,20,20,20,cv.BORDER_REFLECT101)
# wrap=cv.copyMakeBorder(img,20,20,20,20,cv.BORDER_WRAP)
# constant=cv.copyMakeBorder(img,20,20,20,20,cv.BORDER_CONSTANT,value=BLUE)

# plt.subplot(231),plt.imshow(img,'gray'),plt.title('ORIGINAL')
# plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
# plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
# plt.subplot(234),plt.imshow(reflect_201,'gray'),plt.title('REFLECT_201')
# plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
# plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

# plt.show()

img=cv.imread("logoo.jpg")
img1=cv.imread("logo.png")
padded = cv.copyMakeBorder(
    img1,
    top=63,
    bottom=64,
    left=288,
    right=287,
    borderType=cv.BORDER_CONSTANT,  # or BORDER_REFLECT, BORDER_REPLICATE, etc.
    value=[0, 0, 0]  # padding color for each channel (black in this case)
)

# print(img.shape,padded.shape)
# dst=cv.addWeighted(padded,0.3,img,0.7,0)
# cv.imshow("dst",dst)
# cv.waitKey(0)
# cv.destroyAllWindows()
# e1=cv.getTickCount
# x,y,channels =img1.shape
# roi=img[0:x,0:y]
# #create mask
# img12grau=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
# ret,mask=cv.threshold(img12grau,10,255,cv.THRESH_BINARY)
# cv.imshow("mask",mask)
# mask_inv=cv.bitwise_not(mask)

# img_bg=cv.bitwise_and(roi,roi,mask=mask_inv)
# img_fg=cv.bitwise_and(img1,img1,mask=mask)
# dst = cv.add(img_bg,img_fg)
# img[0:x, 0:y ] = dst
# cv.imshow('res',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# e2=cv.getTickCount()
# time = (e2 - e1)/ cv.getTickFrequency()
# print(time)

 
# e1 = cv.getTickCount()
# for i in range(5,49,2):
#     img1 = cv.medianBlur(img1,i)
# e2 = cv.getTickCount()
# t = (e2 - e1)/cv.getTickFrequency()
# print( cv.useOptimized() )


img=cv.imread("face.png")
print(img)
print(img.shape)