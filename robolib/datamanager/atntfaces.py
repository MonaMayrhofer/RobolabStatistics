import os.path
from urllib.request import urlretrieve
import robolib.datamanager.datamanagerutils as du
import zipfile

ATNT_FACE_URL = "http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip"


def get_data(model_name, delete=False):
    dir_name = du.get_model_filename(model_name)
    file_name = dir_name+".zip"
    if delete and os.path.isdir(model_name):
        print("Delete-Mode is on. Deleting directory")
        os.remove(model_name)
    if delete and os.path.isfile(file_name):
        print("Delete-Mode is on. Deleting file")
        os.remove(file_name)
    if os.path.isdir(dir_name):
        print("{} already present, to re-download it remove the directory.".format(dir_name))
    else:
        if os.path.isfile(file_name):
            print("{} already present, to re-download it remove the file.".format(file_name))
        else:
            print("{} not found, downloading it from: {} ".format(file_name, ATNT_FACE_URL))
            urlretrieve(ATNT_FACE_URL, file_name)
        print("File {} is present, extracting it!".format(file_name))
        zip_ref = zipfile.ZipFile(file_name, 'r')
        zip_ref.extractall(dir_name)
        zip_ref.close()
