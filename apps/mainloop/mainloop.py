import cv2
import robolib.modelmanager.downloader as downloader
from robolib.networks.erianet import Erianet
import time

net = Erianet(None, input_to_output_stride=4)
#net.train("3BHIF", 100)
#net.save("3BHIF.model")

MODEL_FILE = 'FrontalFace.xml'
downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_ALT, MODEL_FILE, False)
face_cascades = cv2.CascadeClassifier(MODEL_FILE)

cap = cv2.VideoCapture(0)
facewindows = 0
namelist = []
timeoutlist = []


def get_resized_faces(imgtoresize):
    gray = cv2.cvtColor(imgtoresize, cv2.COLOR_BGR2GRAY)
    faces, rejectlevels, levelleights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, (60, 60), (480, 480), True)
    resfaces = []
    for face in faces:
        x, y, w, h = face
        face = gray[y:y+h, x:x+w]
        resface = cv2.resize(face, dst=None, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
        resfaces.append(resface)
    return resfaces


def show_faces(faces, names):
    for i in range(len(faces)):
        cv2.imshow(names[i], faces[i])
        cv2.waitKey(30)


def recognise_faces(faces):
    names = []
    for face in faces:
        person = net.predict(face, "3BHIF", give_all=True)
        names.append(person[0][0])
        print(person)
    return names


def create_or_destroy_windows(names):
    #Destroying windows of not recognised and timeouted people
    for checkname in namelist:
        exists = False
        index = namelist.index(checkname)
        for name in names:
            if name == checkname:
                exists = True
                timeoutlist[index] = time.time()
        if not exists:
            if time.time() - timeoutlist[index] > 3:
                cv2.destroyWindow(checkname)
                timeoutlist.pop(index)
                namelist.remove(checkname)
    #Creating windows of newly recognised people
    for name in names:
        exists = False
        for checkname in namelist:
            if checkname == name:
                exists = True
        if not exists:
            cv2.namedWindow(name)
            timeoutlist.append(3)
            namelist.append(name)


cv2.namedWindow('img')
while True:
    ret, img = cap.read()
    resizedfaces = get_resized_faces(img)
    recognisednames = recognise_faces(resizedfaces)
    create_or_destroy_windows(recognisednames)
    if len(namelist) != len(resizedfaces):
        print("ERROR: Name count not same as facecount: ")
        print("Names: " + str(namelist))
        print("Facecount: ", len(resizedfaces))
    cv2.imshow('img', img)
    show_faces(resizedfaces, recognisednames)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
