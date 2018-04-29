class PongConfig:
    def __init__(self):
        self.graphics = PongGraphicsConfig()
        self.fps_interval = 0.5
        self.debug = DebugOnConfig()
        self.win_screen_duration = 5
        self.win_screen_times = (0.5, 0.8)


class PongGraphicsConfig:
    def __init__(self):
        self.face_border_thickness = 10
        self.middle_field_brightness = 0.7
        self.win_screen_brightness = 0.4
        self.middle_field_blur = 10
        self.target_cam_fps = 25
        self.fullscreen = False
        self.camera_insets = (0, 0)


class DebugOnConfig:
    def __init__(self):
        self.face_detector_debug = True


CONFIG = PongConfig()
