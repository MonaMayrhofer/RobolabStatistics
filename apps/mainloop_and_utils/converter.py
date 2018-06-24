import cv2
import os
import shutil
import robolib.modelmanager.downloader as downloader
from robolib.util.files import list_dir_recursive
import robolib.datamanager.datadir as datadir
import argparse

MODEL_FILE = downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_ALT, False)
face_cascades = cv2.CascadeClassifier(MODEL_FILE)


def convert_image(source_path, destination_path):
    image = cv2.imread(source_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces, rejectLevels, levelWeights = face_cascades.detectMultiScale3(gray, 1.3, 5, 0, (60, 60), (300, 300), True)
    if len(faces) != 1:
        print(source_path + " could not be converted")
        return False
    x, y, w, h = faces[0]
    if y - 0.22 * h < 0 or y + h * 1.11 > image.shape[0]:
        print(source_path + " could not be converted")
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
        print(destination_path + " does not have enough pictures")
        shutil.rmtree(destination_path)


def main():
    parser = argparse.ArgumentParser(description='Convert faces to suitable erianet-format.')
    parser.add_argument('--src', '-s', type=str, nargs='?',
                        help='The source folder containing the pictures to be converted')
    parser.add_argument('--dst', '-d', type=str, nargs='?',
                        help='The destination folder where the pictures will be converted into')
    parser.add_argument('--exists', '-e', type=str, nargs='?',
                        help='What should happen if dst already exists (O/AO/AA/AI)')
    args = parser.parse_args()
    src = args.src
    dst = args.dst
    mode = args.exists

    if None in [src, dst, mode]:
        print("Please specify arguments. (--help)")
        exit(1)

    if not (mode == 'AO' or mode == 'AA' or mode == 'AI' or mode == 'O'):
        print("Exists invalid")
        exit(1)

    print(src, dst, mode)

    run(src, dst, mode)


def run(inp, out, mode):
    MODE = cv2.COLOR_RGB2GRAY

    src = datadir.get_unconverted_dir(inp)
    if not os.path.isdir(src):
        print("Folder '{0}' does not exist.".format(src))
        exit(1)
    add_overwrite = False
    add_add = False
    add_ignore = False
    if mode == 'AO':
        add_overwrite = True
    elif mode == 'AA':
        add_add = True
    elif mode == 'AI':
        add_ignore = True
    dst = datadir.get_converted_dir(out)
    if os.path.isdir(dst) and not add_overwrite and not add_add and not add_ignore:
        shutil.rmtree(dst)
        os.makedirs(dst)
    elif not os.path.isdir(dst):
        os.makedirs(dst)

    for src_name in os.listdir(src):
        src_path = os.path.join(src, src_name)
        dst_path = os.path.join(dst, src_name)
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
                    while os.path.exists(os.path.join(dst_path, str(img_count) + '.pgm')):
                        img_count += 1
                    if convert_image(os.path.join(src_path, img_name), os.path.join(dst_path, str(img_count) + '.pgm')):
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


if __name__ == "__main__":
    main()
