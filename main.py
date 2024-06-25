import cv2 as cv
import os
import pickle
import face_recognition as fg

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

# Load Encode File
print("Loading Encoded File ...........")
file = open("EncodeFile.p","rb")
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown , studentIds = encodeListKnownWithIds
print(" Encoded File Loaded")

while True:
    success , img = cap.read()
    img = cv.flip(img,1)

    imgS = cv.resize(img , (0,0) , None , 0.25 , 0.25)
    imgS = cv.cvtColor(imgS , cv.COLOR_BGR2RGB)

    faceCurrFrame = fg.face_locations(imgS)
    encodeCurrFrame = fg.face_encodings(imgS,faceCurrFrame)

    imgbackground[162:162+480 , 55:55+640 ] = img
    imgbackground[44:44+633 , 808:808+414 ] = imgModeList[1]

    for encodeFace , faceLoc in zip(encodeCurrFrame,faceCurrFrame):
        matches = fg.compare_faces(encodeListKnown,encodeFace)
        faceDis = fg.face_distance(encodeListKnown,encodeFace)
        print("Matches",matches)
        print("FaceDis",faceDis)

    cv.imshow("Face Attendance", imgbackground)

    cv.waitKey(1)
