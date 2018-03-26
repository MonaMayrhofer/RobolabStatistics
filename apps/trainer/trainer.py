"""
How to use:
Start script.
Enter directory name of person.
If the directory exists, you can either overwrite it or exit the program.
When the 2 Windows appear, check if the grey one reacts to the face.
When pressing 'P' a picture will be taken after the 3 seconds countdown.
With 'S' you can toggle a series of pictures like you press 'P' constantly.
After 10 pictures the program will finish.
The pictures will be named like [1-10].pgm.
The more different the faces are, the better.
"""
import cv2
import time
import robolib.modelmanager.downloader as downloader
import os
import shutil

name = input("Name: ")
if os.path.isdir(name):
    ow = ""
    while ow != "Y" and ow != "N":
        ow = input("Directory exists. Overwrite it? (Y/N): ")
        if ow == "Y":
            shutil.rmtree(name)
        elif ow == "N":
            exit(0)
os.makedirs(name)

MODEL_FILE = 'FrontalFace.xml'
downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_ALT, MODEL_FILE, False)
face_cascades = cv2.CascadeClassifier(MODEL_FILE)
cap = cv2.VideoCapture(0)

imgNumber = 1
lastTime = time.time()

cv2.namedWindow('img')
cv2.namedWindow('resImg')

taking = False
series = False

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, (60, 60), (300, 300), True)
    if len(faces) == 1:
        x, y, w, h = faces[0]
        if int(y - h * 0.2) <= 0 or int(x - w * 0.2) <= 0 or int(y + h * 1.2) >= img.shape[1] or int(x + w * 1.2) >= img.shape[0]:
            print("Outside of camera's range")
        else:
            face = gray[int(y - h * 0.2):int(y + (h * 1.2)), int(x - w * 0.2):int(x + (w * 1.2))]
            resImg = cv2.resize(face, dst=None, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
            if taking and time.time() - lastTime > 3:
                if not series:
                    taking = False
                cv2.imwrite(name + "/" + str(imgNumber) + ".pgm", resImg)
                imgNumber = imgNumber + 1
                lastTime = time.time()
                if imgNumber == 11:
                    break
            if taking:
                cv2.putText(resImg, str(3 - int(time.time() - lastTime)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('resImg', resImg)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if not taking and k == 112:
        lastTime = time.time()
        taking = True
    if k == 115:
        series = not series
        if not taking:
            taking = True
            lastTime = time.time()
cap.release()
cv2.destroyAllWindows()
