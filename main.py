import cv2 as cv
import os
import numpy as np
import cvzone
import pickle
import face_recognition as fg
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import os
import cv2 as cv

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    "databaseURL":"https://facerecognitionrealtime-bd2e2-default-rtdb.firebaseio.com/",
    "storageBucket": "facerecognitionrealtime-bd2e2.appspot.com"
})

cap = cv.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

imgbackground = cv.imread("Resources/background.png")

# Importing the Mode Images to the List
folderPath = "Resources/Modes"
modePathList = os.listdir(folderPath)

imgModeList = []
modetype = 0
counter = 0
id = -1

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
    imgbackground[44:44+633 , 808:808+414 ] = imgModeList[modetype]
    

    for encodeFace , faceLoc in zip(encodeCurrFrame,faceCurrFrame):
        matches = fg.compare_faces(encodeListKnown,encodeFace)
        faceDis = fg.face_distance(encodeListKnown,encodeFace)
        

        matchIndex = np.argmin(faceDis)
        y1 , x2 , y2 , x1 = faceLoc
        y1 , x2 , y2 , x1 = y1*4 , x2*4 , y2*4 , x1*4
        bbox = 55 + x1 , 162 + y1 , x2 - x1 , y2 - y1
        imgbackground = cvzone.cornerRect(imgbackground,bbox,rt = 0,colorC=(0,0,255))

        if matches[matchIndex]:
            id = studentIds[matchIndex]
            imgbackground = cvzone.cornerRect(imgbackground,bbox,rt = 0)
            
            if counter == 0 :
                counter = 1
                modetype = 1

        if not matches[matchIndex]:
            print("Unknown Face")

    if counter != 0:

        if counter == 1:
            studentInfo = db.reference(f"Students/{id}").get()
            print(studentInfo)
        cv.putText(imgbackground , str(studentInfo["Total_Attendance"]),(1005,100),cv.FONT_HERSHEY_COMPLEX,1.5,(255,255,255),2)
        cv.putText(imgbackground , str(id),(938,480),cv.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
        cv.putText(imgbackground , str(studentInfo["Major"]),(960,535),cv.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
        cv.putText(imgbackground , str(studentInfo["Standing"]),(903,645),cv.FONT_HERSHEY_PLAIN,1.5,(193,190,84),2)
        cv.putText(imgbackground , str(studentInfo["Year"]),(1015,645),cv.FONT_HERSHEY_PLAIN,1.5,(193,190,84),2)
        cv.putText(imgbackground , str(studentInfo["Starting_Year"]),(1137,645),cv.FONT_HERSHEY_PLAIN,1.5,(193,190,84),2)
        
        (w,h),_ = cv.getTextSize(studentInfo["Name"],cv.FONT_HERSHEY_COMPLEX,1.2,2)
        offset = (414-w)//2
        cv.putText(imgbackground,str(studentInfo["Name"]),(808+offset,433),cv.FONT_HERSHEY_COMPLEX,1.2,(193,190,84),2)       


        counter += 1
            
            



    cv.imshow("Face Attendance", imgbackground)

    cv.waitKey(1)
