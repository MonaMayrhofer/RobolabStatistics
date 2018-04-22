import cv2
import os

for root, subdirs, files in os.walk("ModelData_AtnTFaces"):
    for file in files:
        if file.endswith(".pgm"):
            full_file = root+"/"+file
            print(full_file)
            img = cv2.imread(full_file, 0)
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
            if not os.path.exists("res_"+root):
                os.makedirs("res_"+root)
            cv2.imwrite("res_"+full_file, img)
