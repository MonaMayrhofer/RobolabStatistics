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
        self.goals_to_win = 1
        self.ready_state_duration = 3


class PongGraphicsConfig:
    def __init__(self):
        self.monitor_size = (1920, 1200)
        self.face_border_thickness = 10
        self.middle_field_brightness = 0.7
        self.win_screen_brightness = 0.4
        self.middle_field_blur = 10
        self.target_cam_fps = 25
        self.fullscreen = False
        self.camera_insets = (0, 0)
        self.goal_font_size = 50
        self.final_screen_font_size = 100


class DebugOnConfig:
    def __init__(self):
        self.face_detector_debug = False


CONFIG = PongConfig()
