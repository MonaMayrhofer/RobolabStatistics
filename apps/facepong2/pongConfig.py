class PongConfig:
    def __init__(self):
        self.graphics = PongGraphicsConfig()
        self.fps_interval = 0.5
        self.debug = DebugOnConfig()
        self.win_screen_duration = 2
        self.win_screen_times = (0.8, 1.6)
        self.final_screen_duration = 2
        self.final_screen_times = (1, 6)
        self.cam = 0
        self.goals_to_win = 3
        self.ready_state_duration = 3
        self.timeout_ready_state_duration = 1
        self.face_missing_timeout = 1.0
        self.max_ball_speed = 320


class PongGraphicsConfig:
    def __init__(self):
        self.monitor_size = (1280, 720)
        self.face_border_thickness = 10
        self.ball_border_thickness = 3
        self.middle_field_brightness = 0.3
        self.win_screen_brightness = 0.4
        self.middle_field_blur = 10
        self.target_cam_fps = 25
        self.fullscreen = True
        self.camera_insets = (0, 0)
        self.goal_font_size = 50
        self.final_screen_font_size = 100
        self.color_face_left_border = (0.0, 0.0, 0.0)
        self.color_face_right_border = (0.0, 0.0, 0.0)
        self.color_ball_border = (0.0, 0.0, 0.0)
        self.color_face_left = (0.5, 1.0, 0.5)
        self.color_face_right = (0.5, 0.5, 1.0)
        self.color_ball = (1.0, 0.5, 0.5)
        self.face_blur = (7, 7)
        self.ball_blur = (5, 5)

class DebugOnConfig:
    def __init__(self):
        self.face_detector_debug = False


CONFIG = PongConfig()
