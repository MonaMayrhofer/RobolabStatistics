import cv2
import os

to_size = (96, 128)
src_folder = "3BHIF"

for root, subdirs, files in os.walk(src_folder):
    for file in files:
        if file.endswith(".pgm"):
            full_file = root+"/"+file
            print(full_file)
            img = cv2.imread(full_file, 0)
            img = cv2.resize(img, to_size, interpolation=cv2.INTER_CUBIC)
            prefix = str(to_size[0])+str(to_size[1])+"res_"
            if not os.path.exists(prefix+root):
                os.makedirs(prefix+root)
            cv2.imwrite(prefix+full_file, img)
