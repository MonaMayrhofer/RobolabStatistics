import sys
import time
from abc import abstractmethod, ABCMeta

import cv2
import numpy as np
import pygame

from apps.facepong2.pongFaceDetector import PongFaceDetector
from apps.facepong2.pongPhysics import PongPhysics
from apps.facepong2.pongRenderer import PongRenderer


class PongGame:
    def __init__(self, file='FrontalFace.xml'):
        self.cap = cv2.VideoCapture(0)
        _, img = self.cap.read()
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.faceDetector = PongFaceDetector(file)
        self.paused = False
        self.winPaused = False
        self.width = img.shape[1]
        self.height = img.shape[0]
        self.third_size = self.width / 3
        self.renderer = PongRenderer((self.width, self.height), (1280, 720))
        self.physics = PongPhysics(self.width, self.height)
        self.state = 0
        self.last_tick = 0
        self.state = ReadyState(2.0)
        self.wins = [0, 0]

    def run(self):
        try:
            self.last_tick = time.time()
            while True:
                img = self.get_image()
                delta = self.update_time()

                self.state.loop(self, img, delta)

                self.renderer.render(img, self, self.state)

                self.handle_events()

        except KeyboardInterrupt:
            pygame.quit()
        except SystemExit:
            pygame.quit()

    def reset(self):
        self.physics.throw_in()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == 27 or event.key == 113:
                    sys.exit(0)
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.dict['size'],
                                                 pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.FULLSCREEN)
                pygame.display.flip()

    def get_image(self):
        _, img = self.cap.read()
        return np.flip(img, 1)

    def get_faces(self, image):
        return self.faceDetector.get_face_positions(image, (30, 30),
                                                    (300, 300), self.third_size, self.third_size * 2)

    def update_faces(self, delta, img):
        right_face, left_face = self.get_faces(img)
        self.physics.faceOne.push_pos(right_face, delta)
        self.physics.faceTwo.push_pos(left_face, delta)
        cnt = 0 if right_face is None else 1
        if left_face is not None:
            cnt += 1
        return cnt

    def update_time(self):
        curr_time = time.time()
        dt = curr_time - self.last_tick
        self.last_tick = curr_time
        return dt

    def win(self, plr):
        self.wins[int((plr+1)/2)] += 1


class GameState(metaclass=ABCMeta):
    @abstractmethod
    def loop(self, game: PongGame, img, delta):
        pass

    @abstractmethod
    def render(self, renderer: PongRenderer, video, game: PongGame):
        pass


class WinState(GameState):
    def render(self, renderer: PongRenderer, video, game: PongGame):
        renderer.draw_background(video)
        renderer.text((0, 0), (255, 255, 255), "Won {0}".format(self.player))

    def __init__(self, player):
        self.time = 3
        self.player = player

    def loop(self, game, img, delta):
        self.time -= delta
        if self.time < 0:
            game.reset()
            game.state = PlayingState()


class ReadyState(GameState):
    def render(self, renderer, video, game: PongGame):
        mat = np.zeros(video.shape, dtype=np.float32)
        mat.fill(0.3)

        if self.face_one is not None:
            cv2.circle(mat, (int(self.face_one[1]), int(self.face_one[0])), 100, (1, 1, 1), -1)

        if self.face_two is not None:
            cv2.circle(mat, (int(self.face_two[1]), int(self.face_two[0])), 100, (1, 1, 1), -1)

        img = cv2.multiply(video, mat, dtype=3)
        renderer.draw_background(img)
        renderer.text((None, None), (255, 255, 255), "{:.1f}".format(self.duration - self.time), size=120)

    def __init__(self, duration):
        self.duration = duration
        self.time = 0
        self.face_one = None
        self.face_two = None

    def loop(self, game, img, delta):
        self.face_one, self.face_two = game.get_faces(img)
        if self.face_one is None or self.face_two is None:
            delta *= -1
        self.time = self.time + delta
        if self.time < 0:
            self.time = 0
        if self.time > self.duration:
            game.state = PlayingState()


class PlayingState(GameState):
    def __init__(self):
        self.timeout = 0

    def render(self, renderer: PongRenderer, video, game: PongGame):
        renderer.draw_background(video)

        ballPos = game.physics.ball.get_pos()
        faceAPos = game.physics.faceOne.get_pos()
        faceBPos = game.physics.faceTwo.get_pos()

        renderer.rect((0, 0, 0), game.width/2, -30, game.width, 30, out=True)
        renderer.text((None, -30), (255, 255, 255), "Wins: {0}-{1}".format(game.wins[0], game.wins[1]), out=True)

        # == Circles ==
        renderer.circle((255, 0, 0),
                        (int(ballPos[0]), int(ballPos[1])), game.physics.ball.radius)
        if faceAPos is not None:
            renderer.circle((0, 255, 0),
                            (int(faceAPos[0]), int(faceAPos[1])), game.physics.faceOne.radius)
        if faceBPos is not None:
            renderer.circle((0, 0, 255),
                            (int(faceBPos[0]), int(faceBPos[1])), game.physics.faceTwo.radius)
        renderer.line((255, 0, 0),
                      (int(video.shape[0] / 3), 0), (int(video.shape[0] / 3), video.shape[1]))
        renderer.line((255, 0, 0), (int(video.shape[0] / 3 * 2), 0),
                      (int(video.shape[0] / 3 * 2), video.shape[1]))

    def loop(self, game, img, delta):
        if game.update_faces(delta, img) < 2:
            self.timeout += delta
        if self.timeout > 1.0:
            game.state = ReadyState(1.0)
        game.physics.tick(delta)
        self.check_state(game)

    def check_state(self, game):
        if game.physics.is_invalid_state():
            game.reset()  # TODO Animation here
        self.check_win(game.physics.get_win(), game)

    def check_win(self, won_player, game):
        if won_player != 0:
            game.win(won_player)
            if won_player < 0:
                game.state = WinState(won_player)
            else:
                game.state = WinState(won_player)

