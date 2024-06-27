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

ref = db.reference('Students')

data = {
    "187080":
    {
        "Name":"Gaurav Haldar",
        "Major":"Mechatronics",
        "Starting_Year":2022,
        "Total_Attendance":34,
        "Standing":"Good",
        "Year":2,
        "Last_Attendance_time":"2024-4-29 00:54:34"
    },
    "268465":
    {
        "Name":"Akshay Kumar",
        "Major":"BMM",
        "Starting_Year":2021,
        "Total_Attendance":45,
        "Standing":"Good",
        "Year":3,
        "Last_Attendance_time":"2024-4-14 00:54:34"
    },
    "458798":
    {
        "Name":"Priyanka Chopra",
        "Major":"Fashion Designing",
        "Starting_Year":2020,
        "Total_Attendance":64,
        "Standing":"Bad",
        "Year":4,
        "Last_Attendance_time":"2023-5-29 00:54:34"
    }
}
for key , value in data.items():
    ref.child(key).set(value)

folderPath = "Images"
PathList = os.listdir(folderPath)

imgList = []
studentIds = []

for path in PathList:
    imgList.append(cv.imread(os.path.join(folderPath,path)))
    studentIds.append(os.path.splitext(path)[0])

    filename = f"{folderPath}/{path}"
    bucket = storage.bucket()
    blob = bucket.blob(filename)
    blob.upload_from_filename(filename)