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

MODEL_FILE = 'FrontalFace.xml'
downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_ALT, MODEL_FILE, False)
face_cascades = cv2.CascadeClassifier(MODEL_FILE)

src = input("Source folder: ")
if not os.path.isdir(src):
    print("Folder does not exist.")
    exit(1)

add_overwrite = False
add_add = False
add_ignore = False


def convert_image(source_path, destination_path):
    image = cv2.imread(source_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, (60, 60), (300, 300), True)
    if len(faces) != 1:
        print(source_path + ' could not be converted')
        return False
    x, y, w, h = faces[0]
    if y - 0.22 * h < 0 or y + h * 1.11 > image.shape[0]:
        print(source_path + ' could not be converted')
        return False
    face = gray[int(y - 0.22 * h):int(y + h * 1.11), x:x + w]
    res_image = cv2.resize(face, dst=None, dsize=(96, 128), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(destination_path, res_image)
    return True


def convert_images(source_path, destination_path):
    image_count = 1
    for image_name in os.listdir(source_path):
        if convert_image(source_path + '/' + image_name,
                         destination_path + '/' + str(image_count) + '.pgm'):
            image_count += 1
    if image_count < 2:
        shutil.rmtree(destination_path)


dst = input("Destination folder: ")
if os.path.isdir(dst):
    ow = ""
    while ow != "O" and ow != "A" and ow != "Q":
        ow = input("Folder already exists. Overwrite folder, add people or exit? (O/A/E): ")
        if ow == "O":
            shutil.rmtree(dst)
            os.makedirs(dst)
        elif ow == "A":
            owa = ""
            while owa != "O" and owa != "A" and owa != "I":
                owa = input("Overwrite existing people, add pictures to people or ignore them? (O/A/I): ")
                if owa == "O":
                    add_overwrite = True
                elif owa == "A":
                    add_add = True
                elif owa == "I":
                    add_ignore = True
        elif ow == "E":
            exit(0)
else:
    os.makedirs(dst)

for src_name in os.listdir(src):
    src_path = src + '/' + src_name
    dst_path = dst + '/' + src_name
    if add_overwrite:
        for dst_name in os.listdir(dst):
            if src_name == dst_name:
                shutil.rmtree(dst_path)
                break
        os.makedirs(dst_path)
        convert_images(src_path, dst_path)
    elif add_add:
        exists = False
        for dst_name in os.listdir(dst):
            if src_name == dst_name:
                exists = True
                break
        if exists:
            img_count = 1
            for img_name in os.listdir(src_path):
                while os.path.exists(dst_path + "/" + str(img_count) + ".pgm"):
                    img_count += 1
                if convert_image(src_path + '/' + img_name, dst_path + '/' + str(img_count) + '.pgm'):
                    img_count += 1
        else:
            os.makedirs(dst_path)
            convert_images(src_path, dst_path)
    elif add_ignore:
        exists = False
        for dst_name in os.listdir(dst):
            if src_name == dst_name:
                exists = True
                break
        if exists:
            continue
        os.makedirs(dst_path)
        convert_images(src_path, dst_path)
    else:
        os.makedirs(dst_path)
        convert_images(src_path, dst_path)
