import cv2 as cv
import face_recognition as fg
import pickle
import os

folderPath = "Images"
PathList = os.listdir(folderPath)

imgList = []
studentIds = []

for path in PathList:
    imgList.append(cv.imread(os.path.join(folderPath,path)))
    studentIds.append(os.path.splitext(path)[0])

def findEncodings(imageList):
    encodeList = []
    for img in imageList:
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        encode = fg.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList 

print("Encoding Started ...........")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown,studentIds]
print("Encoding Completed")

file = open("EncodeFile.p","wb")
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File Saved")