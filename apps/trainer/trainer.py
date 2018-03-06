import cv2
import time
import robolib.modelmanager.downloader as downloader
import os

MODEL_FILE = 'FrontalFace.xml'
downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_ALT, MODEL_FILE, False)
face_cascades = cv2.CascadeClassifier(MODEL_FILE)
cap = cv2.VideoCapture(0)

imgNumber = 1
lastTime = time.time()
name = input("Name: ")
os.makedirs(str(name))

cv2.namedWindow('img')
cv2.namedWindow('resImg')

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, (60, 60), (300, 300), True)
    if len(faces) == 1:
        x, y, w, h = faces[0]
        face = gray[int(y - h * 0.2):int(y + (h * 1.2)), int(x - w * 0.2):int(x + (w * 1.2))]
        resImg = cv2.resize(face, dst=None, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)  # HIER IMST BUGS "imgwarp.cpp:3483: error: (-215) ssize.width > 0 && ssize.height > 0 in function resize"
        if imgNumber == 1 or time.time() - lastTime > 3:
            print("Knips")
            cv2.imwrite(str(name) + "/" + str(imgNumber) + ".pgm", resImg)
            imgNumber = imgNumber + 1
            lastTime = time.time()
            if imgNumber == 11:
                break
        cv2.putText(resImg, str(3 - int(time.time() - lastTime)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('resImg', resImg)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
