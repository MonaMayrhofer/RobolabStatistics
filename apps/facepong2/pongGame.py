from apps.facepong2.pongFaceDetector import PongFaceDetector
import cv2
import numpy as np
from apps.facepong2.pongRenderer import PongRenderer
from apps.facepong2.pongPhysics import PongPhysics
import pygame
import sys
import time


class PongGame:
    def __init__(self, renderer: PongRenderer, file='FrontalFace.xml'):
        self.cap = cv2.VideoCapture(0)
        _, img = self.cap.read()
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.renderer = renderer
        self.faceDetector = PongFaceDetector(file)
        self.paused = False
        self.winPaused = False
        self.physics = PongPhysics(1920, 1280)

    def run(self):
        try:
            last_tick = time.time()
            while True:
                print("Loop")
                # == Calc FPS

                # == Read Image ==
                _, img = self.cap.read()
                img = np.flip(img, 1)
                debug = np.zeros(img.shape)

                field_size = int(img.shape[1] / 3)

                right_face, left_face = self.faceDetector.get_face_positions(img, (30, 30),
                                                                             (300, 300), field_size, field_size*2)

                curr_time = time.time()
                dt = curr_time-last_tick
                if right_face is not None:
                    self.physics.faceOne.push_pos(right_face, dt)
                if left_face is not None:
                    self.physics.faceTwo.push_pos(left_face, dt)
                self.physics.tick(dt)
                last_tick = curr_time
                # == Crop Image ==

                # == Debug Data ==

                # == Show Points
                self.renderer.render(img, self.physics)

                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == 27 or event.key == 113:
                            sys.exit(0)
                    elif event.type == pygame.VIDEORESIZE:
                        screen = pygame.display.set_mode(event.dict['size'],
                                                         pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.FULLSCREEN)
                        pygame.display.flip()

        except KeyboardInterrupt:
            pygame.quit()
        except SystemExit:
            pygame.quit()
