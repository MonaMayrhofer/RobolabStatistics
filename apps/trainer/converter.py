"""
How to use:
Start script.
Enter source folder with subfolders for each person.
Enter destination folder. If it already exists you can add people that have no subfolder in dest.
Every person must have at least 2 valid pictures.
The program will log every picture that did not contain 1 detected face.
"""

import cv2
import os
import shutil
import robolib.modelmanager.downloader as downloader
from robolib.util.files import list_dir_recursive

MODE = cv2.COLOR_RGB2GRAY
MODEL_FILE = 'FrontalFace.xml'
downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_ALT, MODEL_FILE, False)
face_cascades = cv2.CascadeClassifier(MODEL_FILE)

src = input("Source folder: ")
if not os.path.isdir(src):
    print("Folder does not exist.")
    exit(1)

add = False
dst = input("Destination folder: ")
if os.path.isdir(dst):
    ow = ""
    while ow != "O" and ow != "A" and ow != "Q":
        ow = input("Folder already exists. Overwrite, add or quit? (O/A/Q): ")
        if ow == "O":
            shutil.rmtree(dst)
        elif ow == "A":
            add = True
        elif ow == "Q":
            exit(0)
else:
    os.makedirs(dst)

for name in os.listdir(src):
    dstexists = False
    for dstname in os.listdir(dst):
        if name == dstname:
            dstexists = True
    if dstexists or len(os.listdir('./' + src + '/' + name)) < 2:
        continue
    imgcnt = 0
    for imgname in list_dir_recursive(os.path.join(src, name)):
        img = cv2.imread(imgname)
        gray = cv2.cvtColor(img, MODE)
        faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, (60, 60), (300, 300), True)
        if len(faces) != 1:
            print(name + '/' + imgname + ' could not be converted')
            continue
        x, y, w, h = faces[0]
        if y - 0.22 * h < 0 or y + h * 1.11 > img.shape[0]:
            print(name + '/' + imgname + ' could not be converted')
            continue
        imgcnt += 1
        if imgcnt == 1:
            os.makedirs(dst + '/' + name)
        face = gray[int(y - 0.22 * h):int(y + h * 1.11), x:x + w]
        resimg = cv2.resize(face, dst=None, dsize=(96, 128), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(dst + '/' + name + "/" + str(imgcnt) + ".pgm", resimg)
    if imgcnt > 0 and len(os.listdir(dst + '/' + name)) < 2:
        shutil.rmtree(dst + '/' + name)
