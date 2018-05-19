import cv2
import robolib.modelmanager.downloader as downloader
from robolib.networks.erianet import Erianet, ConvolutionalConfig, ClassicConfig, MutliConvConfig
from robolib.networks.common import contrastive_loss_manual
import time
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
# https://www.openu.ac.il/home/hassner/data/lfwa/

data_folder = "conv3BHIF"

print("Using devices: ")
print(device_lib.list_local_devices())

net = Erianet("classcon1_.model", input_image_size=(96, 128), config=ClassicConfig)

MODEL_FILE = 'FrontalFace.xml'
downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_ALT, MODEL_FILE, False)
face_cascades = cv2.CascadeClassifier(MODEL_FILE)

cap = cv2.VideoCapture(0)
#cap.set(3, 1920)
#cap.set(4, 1080)

facewindows = 0
# [0] = name, [1] = timeout for window creation, [2] timeout for window destruction, [3] current probability
personlist = [[], [], [], []]

timeoutin = 3
timeoutout = 8
timeline = dict()


def get_resized_faces(imgtoresize):
    gray = cv2.cvtColor(imgtoresize, cv2.COLOR_BGR2GRAY)
    faces, rejectlevels, levelleights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, (60, 60), (480, 480), True)
    resfaces = []
    for face in faces:
        x, y, w, h = face
        if y - 0.22 * h < 0 or y + h * 1.11 > img.shape[0]:
            continue
        face = gray[int(y - 0.22 * h):int(y + h * 1.11), x:x + w]
        resface = cv2.resize(face, dst=None, dsize=(96, 128), interpolation=cv2.INTER_LINEAR)
        resfaces.append(resface)
    return resfaces


def show_faces(faces, names):
    for i in range(len(faces)):
        if personlist[1][personlist[0].index(names[i])] >= timeoutin:
            print(personlist[3][i])
            cv2.putText(faces[i], str(personlist[3][i]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(names[i], faces[i])
            cv2.waitKey(30)


def recognise_faces(faces):
    names = []
    ts = time.time()
    for face in faces:
        person = net.predict(face, data_folder, give_all=True)
        for i in range(len(personlist[0])):
            if personlist[0][i] == person[0][0]:
                print(person[2])
                personlist[3][i] = person[0][2]
        names.append(person[0][0])
        #print(person)
        #print("Correct: {0} - Incorrect:{0}".format(contrastive_loss_manual(True, person[0][2]),
                                                    #contrastive_loss_manual(False, person[0][2])))
        for name in person:
            if name[0] not in timeline:
                #print(name[0])
                timeline[name[0]] = [[ts], [name[2]]]
            else:
                timeline[name[0]][0].append(ts)
                timeline[name[0]][1].append(name[2])
    return names


def set_timeouts(names):
    # checking for already recognised people
    for i in range(len(personlist[0])):
        exists = False
        for name in names:
            if name == personlist[0][i]:
                exists = True
        # person was recognised
        if exists:
            personlist[1][i] = personlist[1][i] + 1
            # person has not been recognised 3 frames in a row
            if personlist[1][i] > timeoutin:
                personlist[2][i] = timeoutout
        # person was not even recognised 3 frames in a row
        elif personlist[1][i] < timeoutin:
            personlist[0].pop(i)
            personlist[1].pop(i)
            personlist[2].pop(i)
            personlist[3].pop(i)
        # person was not recognised
        else:
            personlist[2][i] = personlist[2][i] - 1
    # checking for newly recognised people
    for name in names:
        exists = False
        for i in range(len(personlist[0])):
            if name == personlist[0][i]:
                exists = True
        if not exists:
            personlist[0].append(name)
            personlist[1].append(1)
            personlist[2].append(0)
            personlist[3].append(0)
    for i in range(len(personlist[0])):
        print("Person: " + personlist[0][i] + ", Timeout in: " + str(personlist[1][i]) + ", Timeout out: " + str(personlist[2][i]))


def create_or_destroy_windows():
    for i in range(len(personlist[1])):
        if personlist[1][i] == timeoutin and personlist[2][i] == 0:
            print("Creating " + personlist[0][i])
            cv2.namedWindow(personlist[0][i])
            personlist[2][i] = timeoutout
        elif personlist[1][i] >= timeoutin and personlist[2][i] == 0:
            print("Destroying " + personlist[0][i])
            cv2.destroyWindow(personlist[0][i])
            personlist[0].pop(i)
            personlist[1].pop(i)
            personlist[2].pop(i)
            personlist[3].pop(i)


cv2.namedWindow('img')
while True:
    ret, img = cap.read()
    resizedfaces = get_resized_faces(img)
    recognisednames = recognise_faces(resizedfaces)
    set_timeouts(recognisednames)
    create_or_destroy_windows()
    if len(recognisednames) != len(resizedfaces):
        print("ERROR: Name count not same as facecount: ")
        print("Names: " + str(personlist[0]))
        print("Facecount: ", len(resizedfaces))
    cv2.imshow('img', img)
    show_faces(resizedfaces, recognisednames)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

legend = []
for key, value in timeline.items():
    plt.plot(value[0], value[1])
    legend.append(key)

plt.legend(legend, loc='upper left')
plt.show()
