import cv2 as cv
import os

cap = cv.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

imgbackground = cv.imread("Resources/background.png")

# Importing the Mode Images to the List
folderPath = "Resources/Modes"
modePathList = os.listdir(folderPath)

imgModeList = []

for path in modePathList:
    imgModeList.append(cv.imread(os.path.join(folderPath,path)))


while True:
    success , img = cap.read()
    img = cv.flip(img,1)

    imgbackground[162:162+480 , 55:55+640 ] = img
    imgbackground[44:44+633 , 808:808+414 ] = imgModeList[1]

    cv.imshow("Face Attendance", imgbackground)

    cv.waitKey(1)
