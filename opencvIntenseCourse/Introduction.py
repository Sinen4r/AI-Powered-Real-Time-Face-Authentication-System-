import cv2 as cv
import sys
import numpy as np
#IMAGE READING SHOWING AND WRITING
# img=cv.imread(cv.samples.findFile("tata.png"))

# if img is None:

#     sys.exit("image not found")
# cv.imshow("image",img)
# k=cv.waitKey(0)
# if k==ord("s"):
    # cv.imwrite("image.png",img)



#VIDEO CAPTURING
# cap=cv.VideoCapture(0)
# if not cap.isOpened() :
#     print("cannot open camera")

# while True:
#     res,frame=cap.read()
#     if not res:
#         print("no frame stream eding?..")
#         break
#     grey=cv.cvtColor(frame,cv.COLOR_RGB2HLS)
#     cv.imshow("frame",grey)
#     if cv.waitKey(1)==ord("q"):
#         break

# cap.release()
# cv.destroyAllWindows()


#PLAYING VIDEO FROM FILE

# cap=cv.VideoCapture("output.mp4")
# while cap.isOpened:
#     res,frame=cap.read()
#     if not res:
#         break
#     frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#     cv.imshow("frame",frame)
#     if cv.waitKey(33)==ord("q"):
#        break
# cv.destroyAllWindows()
# cap.release()

# cap=cv.VideoCapture(0)

# forcc=cv.VideoWriter_fourcc(*"mp4v")
# out=cv.VideoWriter("output.mp4",forcc,20.0,(640,480))
# while cap.isOpened():
#     res,frame=cap.read()
#     frame=cv.flip(frame,1)
#     out.write(frame)
#     cv.imshow("frame",frame)
#     if cv.waitKey(1)==ord("q"):
#         break
# cap.release()
# out.release()
# cv.destroyAllWindows()



#DRAWING FUNCTIONS 
# img=np.zeros((512,512,3),np.uint8)
# cv.line(img,(0,0),(511,511),(255,0,0),5)
# cv.rectangle(img,(384,0),(510,128),(0,255,0),3)

# cv.circle(img,(447,63), 63, (0,0,255), -1)
# cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
# verticies=np.array([[10,5],[20,100],[70,20],[50,10]], np.int32)
# print(verticies.shape)
# verticies=verticies.reshape((-1,1,2))
# cv.polylines(img,[verticies],True,(0,255,255),5)
# font=cv.FONT_HERSHEY_SIMPLEX
# cv.putText(img,"I am going to be the Best AI engineer the world have seen",(0,256),1,1,(255,255,255),2,cv.LINE_AA)
# cv.imshow("frame",img)
# k=cv.waitKey(0)



#HANDLE MOUSE EVENTS
# from random import random

# def drawCircle(event,x,y,flags,param):
#     if event==cv.EVENT_LBUTTONDOWN:
#         cv.circle(img,(x,y),100,(255,0,0),-1)
#         print("circle")

# img=np.zeros((255,255,3),np.uint8)
# cv.namedWindow("window")

# cv.setMouseCallback('window',drawCircle)

# while True:
#     cv.imshow("window",img)
#     if cv.waitKey(20) & 0xFF==27:
#         break
# cv.destroyAllWindows()


# drawing=False  
# mode=True
# ix,iy=-1,-1
# def draw_circle(event,x,y,flags,param):
#     global ix,iy,drawing,mode
#     if event==cv.EVENT_LBUTTONDOWN:
#         drawing=True
#         ix,iy=x,y
#     elif event==cv.EVENT_MOUSEMOVE:
#         if drawing==True:
#             if mode==True:
#                 pass
#             else:
#                 cv.circle(img,(x,y),5,(0,0,255),-1)
#     elif event==cv.EVENT_LBUTTONUP:
#         drawing=False
#         if mode == True:
#             cv.rectangle(img,(ix,iy),(x,y),(0,255,0),2)
#         else:
#             cv.circle(img,(x,y),5,(0,0,255),-1)
# img=np.zeros((255,255,3),np.uint8)
# cv.namedWindow("window")
# cv.setMouseCallback("window",draw_circle)
# while True:
#     cv.imshow("window",img)
#     k=cv.waitKey(2) & 0xFF
#     if k==ord("m"):
#         mode = not mode
#     elif k==27:
#         break
# cv.destroyAllWindows()        
def nothing(x):
    pass

img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('image')
cv.createTrackbar('R','image',0,255,nothing) 
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)

switch='0 : off \n1 : on'

cv.createTrackbar(switch, 'image',0,1,nothing)

while True:
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    r= cv.getTrackbarPos('R','image')
    g = cv.getTrackbarPos('G','image')
    b = cv.getTrackbarPos('B','image')
    s = cv.getTrackbarPos(switch,'image')
    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]