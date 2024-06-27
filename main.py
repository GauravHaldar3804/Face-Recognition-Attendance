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
from datetime import datetime

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
imgStudent = []
bucket = storage.bucket()

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
    
    if faceCurrFrame:
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
                    cvzone.putTextRect(imgbackground,"Loading",(275,480))
                    cv.imshow("Face Attendance", imgbackground)

                    cv.waitKey(1)
                    counter = 1
                    modetype = 1

            if not matches[matchIndex]:
                print("Unknown Face")

        if counter != 0:

            if counter == 1:
                # Get the Data
                
                studentInfo = db.reference(f"Students/{id}").get()
                print(studentInfo)

                # Get the Images from Storage
                blob = bucket.get_blob(f"Images/{id}.png")
                array = np.frombuffer(blob.download_as_string(),np.uint8)
                imgStudent = cv.imdecode(array,cv.COLOR_BGRA2BGR)

                # Update Data of Attendance
                

                #Update data of attendence
                datetimeObject = datetime.strptime(studentInfo['Last_Attendance_time'],"%Y-%m-%d %H:%M:%S")

                secondsElapsed =(datetime.now()-datetimeObject).total_seconds()

                if secondsElapsed > 30:
                    ref = db.reference(f"Students/{id}")
                    studentInfo["Total_Attendance"] += 1
                    ref.child("Total_Attendance").set(studentInfo["Total_Attendance"])
                    ref.child("Last_Attendance_time").set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    print(secondsElapsed)

                else:
                    modetype = 3
                    counter = 0
                    imgbackground[44:44+633 , 808:808+414 ] = imgModeList[modetype]




            if modetype!=3 :   
            
                if 10<counter<20:
                    modetype = 2
                    imgbackground[44:44+633 , 808:808+414 ] = imgModeList[modetype]
                    

                

                if counter <= 10:
                    imgbackground[44:44+633 , 808:808+414 ] = imgModeList[modetype]
                    cv.putText(imgbackground , str(studentInfo["Total_Attendance"]),(1005,100),cv.FONT_HERSHEY_COMPLEX,1.5,(255,255,255),2)
                    cv.putText(imgbackground , str(id),(938,480),cv.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
                    cv.putText(imgbackground , str(studentInfo["Major"]),(960,535),cv.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
                    cv.putText(imgbackground , str(studentInfo["Standing"]),(903,645),cv.FONT_HERSHEY_PLAIN,1.5,(193,190,84),2)
                    cv.putText(imgbackground , str(studentInfo["Year"]),(1015,645),cv.FONT_HERSHEY_PLAIN,1.5,(193,190,84),2)
                    cv.putText(imgbackground , str(studentInfo["Starting_Year"]),(1137,645),cv.FONT_HERSHEY_PLAIN,1.5,(193,190,84),2)

                
                
                    (w,h),_ = cv.getTextSize(studentInfo["Name"],cv.FONT_HERSHEY_COMPLEX,1.2,2)
                    offset = (414-w)//2
                    cv.putText(imgbackground,str(studentInfo["Name"]),(808+offset,433),cv.FONT_HERSHEY_COMPLEX,1.2,(193,190,84),2) 
                    imgbackground[150:150+222 , 905:905+222] = imgStudent
                    


                counter += 1

                if counter>=20:
                    counter = 0
                    modetype = 0
                    studentInfo = []
                    imgStudent = []
                    imgbackground[44:44+633 , 808:808+414 ] = imgModeList[modetype]
    

            
            


    else:
        modetype = 0
        counter = 0


    cv.imshow("Face Attendance", imgbackground)

    cv.waitKey(1)
