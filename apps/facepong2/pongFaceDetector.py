import robolib.modelmanager.downloader as downloader
import cv2
import numpy as np

PONGFACEDETECTORDEBUG = True

class PongFaceDetector:
    def __init__(self, file):
        print()

        downloader.get_model(downloader.HAARCASCADE_FRONTALFACE_ALT, file, False)
        self.face_cascades = cv2.CascadeClassifier(file)

    def get_faces(self, img, min_size, max_size, left_field_end, right_field_end):
        assert left_field_end < right_field_end
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces, rejectLevels, levelWeights = self.face_cascades.detectMultiScale3(
            gray, 1.3, 5, 0, min_size, max_size, True)

        bestleft = None
        bestright = None
        for face in zip(faces, levelWeights):
            mx = face[0][0]+face[0][2]/2
            if mx < left_field_end:
                if bestleft is None or face[1] > bestleft[1]:
                    bestleft = face
            elif mx > right_field_end:
                if bestright is None or face[1] > bestright[1]:
                    bestright = face
        return bestleft, bestright

    def get_face_positions(self, img, min_size, max_size, left_field_end, right_field_end):
        left, right = self.get_faces(img, min_size, max_size, left_field_end, right_field_end)

        left_middle = None if left is None else (left[0][0] + left[0][2]/2, left[0][1] + left[0][3]/2)
        right_middle = None if right is None else (right[0][0] + right[0][2]/2, right[0][1] + right[0][3]/2)

        if PONGFACEDETECTORDEBUG:
            if left_middle is None:
                left_middle = (left_field_end, 10)
            if right_middle is None:
                right_middle = (right_field_end, 10)

        return left_middle, right_middle
