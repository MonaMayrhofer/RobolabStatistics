from apps.facepong2.pongFaceDetector import PongFaceDetector
import cv2
import numpy as np
from apps.facepong2.pongRenderer import PongRenderer
from apps.facepong2.pongPhysics import PongPhysics
import pygame
import sys
import time
from abc import abstractmethod, ABCMeta
from enum import Enum


class GameState(metaclass=ABCMeta):
    @abstractmethod
    def loop(self, game, img, delta):
        pass


class ReadyState(GameState):
    def __init__(self):
        self.time = 0

    def loop(self, game, img, delta):
        self.time = self.time + delta
        print(self.time)
        if self.time > 3:
            game.state = PlayingState()


class PlayingState(GameState):
    def loop(self, game, img, delta):
        right_face, left_face = game.get_faces(img)
        game.physics.faceOne.push_pos(right_face, delta)
        game.physics.faceTwo.push_pos(left_face, delta)
        game.physics.tick(delta)


class PongGame:
    def __init__(self, renderer: PongRenderer, file='FrontalFace.xml'):
        self.cap = cv2.VideoCapture(0)
        _, img = self.cap.read()
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.renderer = renderer
        self.faceDetector = PongFaceDetector(file)
        self.paused = False
        self.winPaused = False
        self.width = img.shape[1]
        self.height = img.shape[0]
        self.third_size = self.width / 3
        self.physics = PongPhysics(self.width, self.height)
        self.state = 0
        self.last_tick = 0
        self.state = ReadyState()

    def run(self):
        try:
            self.last_tick = time.time()
            while True:
                print("Loop")

                img = self.get_image()
                delta = self.update_time()

                self.state.loop(self, img, delta)

                self.renderer.render(img, self.physics, self.state)

                self.handle_events()

        except KeyboardInterrupt:
            pygame.quit()
        except SystemExit:
            pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == 27 or event.key == 113:
                    sys.exit(0)
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.dict['size'],
                                                 pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.FULLSCREEN)
                pygame.display.flip()

    def check_state(self):
        if self.physics.is_invalid_state():
            self.reset()
        self.check_win(self.physics.get_win())

    def reset(self):
        self.physics.throw_in()

    def check_win(self, won_player):
        if won_player == 0:
            return

    def get_image(self):
        _, img = self.cap.read()
        return np.flip(img, 1)

    def get_faces(self, image):
        return self.faceDetector.get_face_positions(image, (30, 30),
                                                    (300, 300), self.third_size, self.third_size * 2)

    def update_time(self):
        curr_time = time.time()
        dt = curr_time - self.last_tick
        self.last_tick = curr_time
        return dt
