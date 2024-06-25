import cv2 as cv

cap = cv.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

imgbackground = cv.imread("Resources/background.png")

while True:
    success , img = cap.read()
    cv.imshow("Webcam", img)
    cv.imshow("Face Attendance", imgbackground)

    cv.waitKey(1)
