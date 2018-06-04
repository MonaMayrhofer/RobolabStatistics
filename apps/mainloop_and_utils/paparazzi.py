import cv2
import time
import robolib.modelmanager.downloader as downloader
import os
import shutil
import robolib.datamanager.datadir as datadir

print(datadir.__DATA_DIR)

group = input("Group: ")
name = input("Name: ")
imdir = datadir.paparazzi_output(name, group)
if os.path.isdir(imdir):
    ow = ""
    while ow != "O" and ow != "A" and ow != "E":
        ow = input("Folder '"+imdir+"' already exists. Overwrite folder, add pictures or exit? (O/A/E): ")
        if ow == "O":
            shutil.rmtree(imdir)
            os.makedirs(imdir)
        elif ow == "E":

            exit(0)
else:
    os.makedirs(imdir)
print("Take pictures with P and start picture series with S. Exit with ESC")

MODEL_FILE = 'FrontalFace.xml'
downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_ALT, MODEL_FILE, False)
face_cascades = cv2.CascadeClassifier(MODEL_FILE)
cap = cv2.VideoCapture(0)
#cap.set(3, 1920)
#cap.set(4, 1080)

image_number = 1
while os.path.exists(os.path.join(imdir, name, str(image_number) + ".pgm")):
    image_number += 1
last_time = time.time()

cv2.namedWindow('img')
cv2.namedWindow('res_img')

taking = False
series = False

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, (60, 60), (300, 300), True)
    if len(faces) == 1:
        x, y, w, h = faces[0]
        if not (y - 0.22 * h < 0 or y + h * 1.11 > img.shape[0]):
            face = gray[int(y - 0.22 * h):int(y + h * 1.11), x:x + w]
            res_img = cv2.resize(face, dst=None, dsize=(96, 128), interpolation=cv2.INTER_LINEAR)
            if taking and time.time() - last_time > 3:
                if not series:
                    taking = False
                cv2.imwrite(os.path.join(imdir, str(image_number) + ".pgm"), res_img)
                print("Picture taken")
                while os.path.exists(os.path.join(imdir, str(image_number) + ".pgm")):
                    image_number += 1
            last_time = time.time()
            if taking:
                cv2.putText(res_img, str(3 - int(time.time() - last_time)), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('res_img', res_img)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if not taking and k == 112:
        last_time = time.time() - 3
        taking = True
    if k == 115:
        series = not series
        if not taking:
            taking = True
            last_time = time.time()
cap.release()
cv2.destroyAllWindows()
