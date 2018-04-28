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
        self.fps = 0.0
        self.last_img_time = 0
        self.last_img = None
        self.target_cam_fps = 24  # TODO Optimize this
        self.last_fps_print = 0

    def run(self):
        try:
            self.last_tick = time.time()
            while True:
                self.fps = 1/(time.time() - self.last_tick)
                if time.time() - self.last_fps_print > 0.5:
                    self.last_fps_print = time.time()
                    print("{0:.2f} Fps".format(self.fps))
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
        if self.last_img is None or time.time() - self.last_img_time > (1/self.target_cam_fps):
            _, self.last_img = self.cap.read()
            self.last_img = np.flip(self.last_img, 1)
            self.last_img_time = time.time()
        return self.last_img

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
        self.wins[int((plr + 1) / 2)] += 1


class GameState(metaclass=ABCMeta):
    @abstractmethod
    def loop(self, game: PongGame, img, delta):
        pass

    @abstractmethod
    def render(self, renderer: PongRenderer, video, game: PongGame):
        pass


class ReadyState(GameState):
    def render(self, renderer, video, game: PongGame):
        mat = np.zeros(video.shape, dtype=np.float32)
        mat.fill(0.0)

        if self.face_one is not None:
            cv2.circle(mat, (int(self.face_one[1]), int(self.face_one[0])), 100, (1, 1, 1), -1)

        if self.face_two is not None:
            cv2.circle(mat, (int(self.face_two[1]), int(self.face_two[0])), 100, (1, 1, 1), -1)

        mat = cv2.blur(mat, (20, 20))
        img = cv2.multiply(cv2.cvtColor(video, cv2.COLOR_RGB2GRAY), 0.5)
        img = cv2.blur(img, (50, 50))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.multiply(img, 1-mat, dtype=3)
        video = cv2.multiply(video, mat, dtype=3)
        img = cv2.add(img, video)
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
        self.draw_field(renderer, video, game)

        #Balls
        ballPos = game.physics.ball.get_pos()
        faceAPos = game.physics.faceOne.get_pos()
        faceBPos = game.physics.faceTwo.get_pos()

        # == Circles ==
        renderer.circle((244, 120, 30),
                        (int(ballPos[0]), int(ballPos[1])), game.physics.ball.radius)
        if faceAPos is not None:
            renderer.circle((0, 255, 0),
                            (int(faceAPos[0]), int(faceAPos[1])), game.physics.faceOne.radius)
        if faceBPos is not None:
            renderer.circle((0, 0, 255),
                            (int(faceBPos[0]), int(faceBPos[1])), game.physics.faceTwo.radius)

        #Goals
        renderer.rect((0, 0, 0), game.width / 2, -30, game.width, 30, out=True)
        renderer.text((None, -30), (255, 255, 255), "Wins: {0}-{1}".format(game.wins[0], game.wins[1]), out=True)

    def draw_field(self, renderer: PongRenderer, video, game: PongGame):
        #Middlefield
        middlemat = np.zeros(video.shape, dtype=np.float32)
        cv2.rectangle(middlemat, (0, int(video.shape[0] / 3)), (video.shape[1], int(video.shape[0]/3*2)), (1.0, 1, 1),
                      thickness=-1)
        middlemat = cv2.blur(middlemat, (20, 20))

        middlefieldvid = cv2.blur(video, (50, 50))
        middlefieldvid = cv2.multiply(middlefieldvid, 0.6*middlemat, dtype=3)
        restvid = cv2.multiply(video, 1-middlemat, dtype=3)
        renderer.draw_background(cv2.add(middlefieldvid, restvid))

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


class WinState(PlayingState):
    def render(self, renderer: PongRenderer, video, game: PongGame):
        middlemat = np.zeros(video.shape, dtype=np.float32)
        cv2.rectangle(middlemat, (0, int(video.shape[0] / 3)), (video.shape[1], int(video.shape[0]/3*2)), (1.0, 1, 1),
                      thickness=-1)
        middlemat = cv2.blur(middlemat, (20, 20))

        middlefieldvid = cv2.blur(video, (50, 50))
        middlefieldvid = cv2.multiply(middlefieldvid, 0.6*middlemat, dtype=3)
        restvid = cv2.multiply(video, 1-middlemat, dtype=3)
        renderer.draw_background(cv2.add(middlefieldvid, restvid))

    def __init__(self, player):
        super().__init__()
        self.time = 3
        self.player = player

    def loop(self, game, img, delta):
        self.time -= delta
        if self.time < 0:
            game.reset()
            game.state = PlayingState()