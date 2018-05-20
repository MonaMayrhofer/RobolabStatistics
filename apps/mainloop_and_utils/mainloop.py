import cv2
import robolib.modelmanager.downloader as downloader
from robolib.networks.erianet import Erianet
from robolib.networks.configurations import VGG19ish
from robolib.networks.common import contrastive_loss_manual
import time
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
from robolib.networks.predict_result import PredictResult
# https://www.openu.ac.il/home/hassner/data/lfwa/


class PersonData:
    def __init__(self, name):
        self.name = name
        self.timeout_in = 1
        self.timeout_out = 0
        self.probability = 0

    # Returns False if person was not even recognised 3 frames in a row
    def recognised(self, recognised):
        if recognised:
            if self.timeout_in < 4:
                self.timeout_in += 1
                if self.timeout_in == 3:
                    self.timeout_out = 8
            else:
                self.timeout_out = 8
        else:
            if self.timeout_in < 3:
                return False
            else:
                self.timeout_out -= 1
        return True


data_folder = "conv3BHIF"
data_folder = "intermconv3BHIFbigset"

print("Using devices: ")
print(device_lib.list_local_devices())

net = Erianet("bigset_4400_1526739422044.model", input_image_size=(96, 128), config=VGG19ish)

MODEL_FILE = 'FrontalFace.xml'
downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_ALT, MODEL_FILE, False)
face_cascades = cv2.CascadeClassifier(MODEL_FILE)

cap = cv2.VideoCapture(0)
#cap.set(3, 1920)
#cap.set(4, 1080)

# [0] = name, [1] = timeout for window creation, [2] timeout for window destruction, [3] current probability
person_list = []

timeout_in = 3
timeout_out = 8
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
    for person in person_list:
        for name in names:
            if person.name == name and person.timeout_in >= timeout_in:
                cv2.putText(faces[names.index(name)], str(person.probability), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow(name, faces[names.index(name)])
                cv2.waitKey(30)


def recognise_faces(faces):
    names = []
    ts = time.time()
    for face in faces:
        predicted_person = net.predict(face, data_folder, give_all=True)
        for person in person_list:
            if person.name == predicted_person[0][0]:
                person.probability = predicted_person[0][2]
        names.append(predicted_person[0][0])
        #print(predicted_person)
        #print("Correct: {0} - Incorrect:{0}".format(contrastive_loss_manual(True, predicted_person[0][2]),
                                                    #contrastive_loss_manual(False, predicted_person[0][2])))
        for name in predicted_person:
            if name[0] not in timeline:
                #print(name[0])
                timeline[PredictResult.name(name)] = [[ts], [PredictResult.difference(name())]]
            else:
                timeline[name[0]][0].append(ts)
                timeline[name[0]][1].append(name[1])
    return names


def set_timeouts(names):
    # checking for already recognised people
    for person in reversed(person_list):
        exists = False
        for name in names:
            if name == person.name:
                exists = True
        if not person.recognised(exists):
            person_list.remove(person)
    # checking for newly recognised people
    for name in names:
        exists = False
        for person in person_list:
            if name == person.name:
                exists = True
        if not exists:
            person_list.append(PersonData(name))
    for person in person_list:
        print("Person: " + person.name + ", Timeout in: " + str(person.timeout_in) + ", Timeout out: " + str(person.timeout_out))


def create_or_destroy_windows():
    for person in reversed(person_list):
        if person.timeout_in == timeout_in and person.timeout_out == timeout_out:
            print("Creating " + person.name)
            cv2.namedWindow(person.name)
        elif person.timeout_in >= timeout_in and person.timeout_out == 0:
            print("Destroying " + person.name)
            cv2.destroyWindow(person.name)
            person_list.remove(person)


cv2.namedWindow('img')
while True:
    ret, img = cap.read()
    print("Image read")
    resized_faces = get_resized_faces(img)
    recognised_names = recognise_faces(resized_faces)
    set_timeouts(recognised_names)
    create_or_destroy_windows()
    if len(recognised_names) != len(resized_faces):
        print("ERROR: Name count not same as facecount: ")
        print("Names: " + str(person_list[0]))
        print("Facecount: ", len(resized_faces))
    cv2.imshow('img', img)
    show_faces(resized_faces, recognised_names)
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
