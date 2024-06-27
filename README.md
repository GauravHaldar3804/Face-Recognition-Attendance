# Face Recognition Attendance System

This project implements a face recognition system for marking student attendance.

## Files

* **main.py:**
  * Connects to Firebase for student data.
  * Captures video frames from the webcam.
  * Detects faces and compares them to known encodings.
  * Displays student information and updates attendance records.
  * Uses multiple images for different attendance states.
* **AddDataTodatabase.py:**
  * Adds student data (name, major, attendance) to Firebase Realtime Database.
  * Uploads student images to Firebase Storage.
* **EncodeGenerator.py:**
  * Generates a file containing encoded facial representations of students.
  * Reads student images from a folder.
  * Uses `face_recognition` to encode faces.
  * Saves encoded faces and student IDs to a pickle file (.p) for `main.py`.

## Dependencies

* OpenCV (cv2)
* face_recognition (fg)
* pickle
* numpy (np)
* cvzone (optional for text rectangles)
* firebase_admin

## Setup

1. Install dependencies: `pip install opencv-python face-recognition pickle numpy cvzone firebase-admin`
2. Create a Firebase project and enable Realtime Database and Storage.
3. Download the service account key file.
4. Place the key file in your project directory as `serviceAccountKey.json`.
5. Create folders named "Images" and "Resources" in your project directory.
6. Add student images to "Images" and attendance state images to "Resources/Modes".
7. Run `AddDataTodatabase.py` to add student data and upload images.
8. Run `EncodeGenerator.py` to generate the encoded faces file (`EncodeFile.p`).
9. Run `main.py` to start the application.

git
