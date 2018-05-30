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
import robolib.datamanager.datadir as datadir
import argparse


def main():
    parser = argparse.ArgumentParser(description='Convert faces to suitable erianet-format.')
    parser.add_argument('--input', '-i', type=str, nargs='?',
                        help='Folder in data-directory that needs to be converted.')
    parser.add_argument('--output', '-o', type=str, nargs='?',
                        help='Destination folder in data-directory that will hold the converted images.')

    args = parser.parse_args()
    inp = args.input
    out = args.output

    if None in [inp, out]:
        print("Please specify arguments. (--help)")
        exit(1)

    print(inp, out)

    run(inp, out)


def run(inp, out):
    MODE = cv2.COLOR_RGB2GRAY
    MODEL_FILE = downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_ALT, False)
    face_cascades = cv2.CascadeClassifier(MODEL_FILE)

    src = datadir.get_unconverted_dir(inp)
    if not os.path.isdir(src):
        print("Folder '{0}' does not exist.".format(src))
        exit(1)

    add = False
    dst = datadir.get_converted_dir(out)
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
        dst_exists = False
        for dst_name in os.listdir(dst):
            if name == dst_name:
                dst_exists = True
        if dst_exists or len(os.listdir(os.path.join(src, name))) < 2:
            continue
        img_cnt = 0
        for img_name in list_dir_recursive(os.path.join(src, name)):
            img = cv2.imread(img_name)
            gray = cv2.cvtColor(img, MODE)
            faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0,
                                                                                (60, 60), (300, 300), True)
            if len(faces) != 1:
                print('{0} could not be converted'.format(img_name))
                continue
            x, y, w, h = faces[0]
            if y - 0.22 * h < 0 or y + h * 1.11 > img.shape[0]:
                print('{0} could not be converted'.format(img_name))
                continue
            img_cnt += 1
            if img_cnt == 1:
                os.makedirs(os.path.join(dst, name))
            face = gray[int(y - 0.22 * h):int(y + h * 1.11), x:x + w]
            resimg = cv2.resize(face, dsize=(96, 128), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(dst, name, str(img_cnt) + ".pgm"), resimg)
        if img_cnt > 0 and len(os.listdir(os.path.join(dst, name))) < 2:
            shutil.rmtree(os.path.join(dst, name))


if __name__ == "__main__":
    main()
