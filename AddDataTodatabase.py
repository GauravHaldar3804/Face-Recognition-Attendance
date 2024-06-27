import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    "databaseURL":"https://facerecognitionrealtime-bd2e2-default-rtdb.firebaseio.com/"
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