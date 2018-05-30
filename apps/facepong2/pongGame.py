import sys
import time
from abc import abstractmethod, ABCMeta

import cv2
import numpy as np
import pygame

from apps.facepong2.pongFaceDetector import PongFaceDetector
from apps.facepong2.pongPhysics import PongPhysics
from apps.facepong2.pongConfig import CONFIG
from apps.facepong2.pongRenderer import PongRenderer, TextAlign


class PongGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(CONFIG.cam)
        #self.cap.set(3, 1920)
        #self.cap.set(4, 1080)
        _, img = self.cap.read()
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.faceDetector = PongFaceDetector()
        self.width = img.shape[1]
        self.height = img.shape[0]
        self.third_size = self.width / 3
        self.renderer = PongRenderer((self.width, self.height), CONFIG.graphics.monitor_size)
        self.physics = PongPhysics(self.width, self.height)
        self.state = ReadyState()
        self.fps = 0.0
        self.last_img_time = 0
        self.last_img = None
        self.last_fps_print = 0
        self.last_tick = 0

        # Restart
        self.paused = False
        self.winPaused = False
        self.wins = [0, 0]
        self.restart()

    def restart(self):
        self.paused = False
        self.winPaused = False
        self.wins = [0, 0]

    def run(self):
        try:
            self.last_tick = time.time()
            while True:
                self.fps = 1 / max(time.time() - self.last_tick, 0.000001)
                if time.time() - self.last_fps_print > CONFIG.fps_interval:
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
                # TODO What did I want do do here?
                # screen = pygame.display.set_mode(event.dict['size'],
                #                                 pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.FULLSCREEN)
                pygame.display.flip()

    def get_image(self):
        if self.last_img is None or time.time() - self.last_img_time > (1 / CONFIG.graphics.target_cam_fps):
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

    def wins_of(self, plr):
        return self.wins[int((plr + 1) / 2)]
    
    def change_state_to(self, state):
        #print("Changing state...")
        #print(state)
        self.state = state


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
        img = cv2.multiply(img, 1 - mat, dtype=3)
        video = cv2.multiply(video, mat, dtype=3)
        img = cv2.add(img, video)
        renderer.draw_background(img)
        renderer.text(None, (255, 255, 255), "{:.1f}s".format(self.duration - self.time), size=120, out=True)

    def __init__(self, duration=None, reset=True):
        if duration is None:
            duration = CONFIG.ready_state_duration
        self.duration = duration
        self.time = 0
        self.face_one = None
        self.face_two = None
        self.reset = reset

    def loop(self, game, img, delta):
        self.face_one, self.face_two = game.get_faces(img)
        if self.face_one is None or self.face_two is None:
            delta *= -1
        self.time = self.time + delta
        if self.time < 0:
            self.time = 0
        if self.time > self.duration:
            if self.reset:
                game.reset()
            game.change_state_to(PlayingState())


class PlayingState(GameState):
    def __init__(self):
        self.timeout = 0

    @staticmethod
    def get_blurred_field(video, upper_left, lower_right, blur, brightness):
        middle_mat = np.zeros(video.shape, dtype=np.float32)
        cv2.rectangle(middle_mat, upper_left, lower_right, (1.0, 1, 1), thickness=-1)
        middle_mat = cv2.blur(middle_mat, (20, 20))

        middle_field_vid = cv2.blur(video, (int(blur), int(blur)))
        middle_field_vid = cv2.multiply(middle_field_vid, brightness * middle_mat, dtype=3)
        rest_vid = cv2.multiply(video, 1 - middle_mat, dtype=3)

        return cv2.add(middle_field_vid, rest_vid)

    def draw_points(self, renderer: PongRenderer, y, points_a, points_b, alpha=255):
        # Goals
        renderer.text((None, y), (255, 255, 255, alpha), "{0} - {1}".format(points_a, points_b), out=True,
                      align=(TextAlign.CENTER, TextAlign.LEFT), size=50, draw_in_front=False)

    def render(self, renderer: PongRenderer, video, game: PongGame):
        # Middlefield
        background = self.get_blurred_field(video,
                                            (0, int(video.shape[0] / 3)), (video.shape[1], int(video.shape[0] / 3 * 2)),
                                            CONFIG.graphics.middle_field_blur, CONFIG.graphics.middle_field_brightness)

        # Faces
        face_mat = np.zeros(video.shape, dtype=np.float32)
        face_mat.fill(1.0)
        face_a_pos = game.physics.faceOne.get_pos()
        face_b_pos = game.physics.faceTwo.get_pos()
        thickness = CONFIG.graphics.face_border_thickness

        if face_a_pos is not None:
            cv2.circle(face_mat, (int(face_a_pos[1]), int(face_a_pos[0])),
                       game.physics.faceOne.radius + thickness, CONFIG.graphics.color_face_left_border, -1)
            cv2.circle(face_mat, (int(face_a_pos[1]), int(face_a_pos[0])),
                       game.physics.faceOne.radius, CONFIG.graphics.color_face_left, -1)
        if face_b_pos is not None:
            cv2.circle(face_mat, (int(face_b_pos[1]), int(face_b_pos[0])),
                       game.physics.faceOne.radius + thickness, CONFIG.graphics.color_face_right_border, -1)
            cv2.circle(face_mat, (int(face_b_pos[1]), int(face_b_pos[0])),
                       game.physics.faceOne.radius, CONFIG.graphics.color_face_right, -1)
        face_mat = cv2.blur(face_mat, CONFIG.graphics.face_blur)
        ball_pos = game.physics.ball.get_pos()
        cv2.circle(face_mat, (int(ball_pos[1]), int(ball_pos[0])),
                   game.physics.ball.radius+CONFIG.graphics.ball_border_thickness,
                   CONFIG.graphics.color_ball_border, -1)
        cv2.circle(face_mat, (int(ball_pos[1]), int(ball_pos[0])), game.physics.ball.radius,
                   CONFIG.graphics.color_ball, -1)
        face_mat = cv2.blur(face_mat, CONFIG.graphics.ball_blur)
        background = cv2.multiply(background, face_mat, dtype=3)

        renderer.draw_background(background)

        # Ball
        #ball_pos = game.physics.ball.get_pos()
        #renderer.circle((0, 0, 30),
        #                (int(ball_pos[0]), int(ball_pos[1])), game.physics.ball.radius)

        self.draw_points(renderer, -CONFIG.graphics.goal_font_size, game.wins[0], game.wins[1])

    def loop(self, game, img, delta):
        if game.update_faces(delta, img) < 2:
            self.timeout += delta
        if self.timeout > CONFIG.face_missing_timeout:
            game.change_state_to(ReadyState(CONFIG.timeout_ready_state_duration, False))
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
                game.change_state_to(WinState(won_player))
            else:
                game.change_state_to(WinState(won_player))


class WinState(PlayingState):
    def render(self, renderer: PongRenderer, video, game: PongGame):
        #print(game.wins_of(self.player))
        if game.wins_of(self.player) >= CONFIG.goals_to_win:
            game.change_state_to(FinalState(self.player))
            return

        # Timing
        time_progress = min(1.0, (self.duration - self.time) / self.duration)

        fade_in_end = CONFIG.win_screen_times[0] / CONFIG.win_screen_duration
        fade_out_start = CONFIG.win_screen_times[1] / CONFIG.win_screen_duration
        time_progress = time_progress / fade_in_end if time_progress < fade_in_end else (
            1 if time_progress < fade_out_start else
            (1 - time_progress) / (1 - fade_out_start)
        )

        # Background-Animation
        player_id = (self.player + 1) / 2
        a_third = video.shape[0] / 3
        x_start = (1 - time_progress * player_id) * a_third
        x_end = video.shape[0] - ((1 - time_progress * (1 - player_id)) * a_third)
        blur = time_progress * CONFIG.graphics.middle_field_blur * 4 + CONFIG.graphics.middle_field_blur
        brightness = CONFIG.graphics.middle_field_brightness - (time_progress * CONFIG.graphics.middle_field_brightness
                                                                * (1 - CONFIG.graphics.win_screen_brightness))
        background = self.get_blurred_field(video, (0, int(x_start)), (video.shape[1], int(x_end)), blur, brightness)
        renderer.draw_background(background)

        # Texts
        text_x = a_third + player_id * a_third - self.player * 30
        text_align = TextAlign.RIGHT if self.player > 0 else TextAlign.LEFT
        lr_text = "Left" if self.player < 0 else "Right"
        text_alpha = max(0.0, time_progress * 355 - 100)
        renderer.text((text_x, None), (255, 255, 255, text_alpha), "{0} Goal".format(lr_text), size=60,
                      align=(text_align, TextAlign.CENTER), out=True)

        # Goals Text
        win_a_pre = int(game.wins[0] - (1 - player_id))
        win_b_pre = int(game.wins[1] - player_id)
        if self.time > self.duration * fade_in_end:
            self.draw_points(renderer, -CONFIG.graphics.goal_font_size * (1 - time_progress), win_a_pre, win_b_pre,
                             alpha=255 * (1 - time_progress))
        else:
            self.draw_points(renderer, -CONFIG.graphics.goal_font_size * (1 - time_progress), game.wins[0],
                             game.wins[1],
                             alpha=255 * (1 - time_progress))

    def __init__(self, player):
        super().__init__()
        self.duration = CONFIG.win_screen_duration
        self.time = self.duration
        self.player = player

    def loop(self, game, img, delta):
        self.time -= delta
        if self.time < 0:
            game.reset()
            game.change_state_to(PlayingState())


class FinalState(PlayingState):
    def __init__(self, won_player):
        super().__init__()
        self.player = won_player
        self.time = 0
        self.duration = CONFIG.final_screen_duration

    def render(self, renderer: PongRenderer, video, game: PongGame):
        # Timing
        time_progress = min(1.0, self.time / CONFIG.final_screen_times[0])

        # Background-Animation
        player_id = (self.player + 1) / 2
        a_third = video.shape[0] / 3
        x_start = (1 - time_progress) * a_third
        x_end = video.shape[0] - ((1 - time_progress) * a_third)
        blur = time_progress * CONFIG.graphics.middle_field_blur * 4 + CONFIG.graphics.middle_field_blur
        brightness = CONFIG.graphics.middle_field_brightness - (time_progress * CONFIG.graphics.middle_field_brightness
                                                                * (1 - CONFIG.graphics.win_screen_brightness))
        background = self.get_blurred_field(video, (0, int(x_start)), (video.shape[1], int(x_end)), blur, brightness)
        renderer.draw_background(background)

        # Points
        pointsY = -CONFIG.graphics.goal_font_size \
                  + (CONFIG.graphics.goal_font_size + renderer.screen.get_height() / 2
                  - CONFIG.graphics.final_screen_font_size / 2) * time_progress
        renderer.text((None, pointsY), (255, 255, 255), "{0} - {1}".format(game.wins[0], game.wins[1]), out=True,
                      align=(TextAlign.CENTER, TextAlign.LEFT),
                      size=int(CONFIG.graphics.goal_font_size +
                               (
                                           CONFIG.graphics.final_screen_font_size - CONFIG.graphics.goal_font_size) * time_progress))

    def loop(self, game, img, delta):
        self.time += delta
        if self.time >= self.duration:
            game.restart()
            game.change_state_to(ReadyState())