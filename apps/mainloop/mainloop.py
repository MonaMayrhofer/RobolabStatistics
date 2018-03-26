import cv2
import numpy as np
import robolib.modelmanager.downloader as downloader

MODEL_FILE = 'FrontalFace.xml'
downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_ALT, MODEL_FILE, False)
face_cascades = cv2.CascadeClassifier(MODEL_FILE)

cap = cv2.VideoCapture(0)
facewindows = 0


def get_resized_faces(imgtoresize):
    gray = cv2.cvtColor(imgtoresize, cv2.COLOR_BGR2GRAY)
    faces, rejectlevels, levelleights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, (60, 60), (480, 480), True)
    facecount = 0
    for _ in faces:
        x, y, w, h = faces[0]
        if not (int(y - h * 0.2) <= 0 or int(x - w * 0.2) <= 0 or int(y + h * 1.2) >= imgtoresize.shape[1] or int(x + w * 1.2) >= imgtoresize.shape[0]):
            facecount += 1
    print(facecount)
    resfaces = np.zeros((facecount, 128, 128), dtype=np.uint8)
    index = 0
    for _ in faces:
        x, y, w, h = faces[0]
        if int(y - h * 0.2) <= 0 or int(x - w * 0.2) <= 0 or int(y + h * 1.2) >= imgtoresize.shape[1] or int(x + w * 1.2) >= imgtoresize.shape[0]:
            continue
        face = gray[int(y - h * 0.2):int(y + (h * 1.2)), int(x - w * 0.2):int(x + (w * 1.2))]
        resimg = cv2.resize(face, dst=None, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
        resfaces[index] = resimg
        index += 1
    return resfaces


def show_faces(facestoshow):
    name = 0
    for facetoshow in facestoshow:
        print(facetoshow)
        cv2.imshow(str(name), facetoshow)
        name += 1
        cv2.waitKey(30)


def recognise_faces(facestorecognise):
    return


cv2.namedWindow('img')
while True:
    ret, img = cap.read()
    resizedfaces = get_resized_faces(img)
    if facewindows > len(resizedfaces):
        for i in range(facewindows - len(resizedfaces)):
            print('Destroyed window ' + str(len(resizedfaces) + i))
            cv2.destroyWindow(str(len(resizedfaces) + i))
    elif facewindows < len(resizedfaces):
        for i in range(len(resizedfaces) - facewindows):
            print('Created window ' + str(facewindows + i))
            cv2.namedWindow(str(facewindows + i))
    facewindows = len(resizedfaces)
    cv2.imshow('img', img)
    show_faces(resizedfaces)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
