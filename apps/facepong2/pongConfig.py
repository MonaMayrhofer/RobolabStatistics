class PongConfig:
    def __init__(self):
        self.graphics = PongGraphicsConfig()
        self.fps_interval = 0.5
        self.debug = DebugOnConfig()


class PongGraphicsConfig:
    def __init__(self):
        self.face_border_thickness = 10
        self.middle_field_brightness = 0.7
        self.middle_field_blur = 10
        self.target_cam_fps = 25
        self.fullscreen = False
        self.camera_insets = (0, 0)


class DebugOnConfig:
    def __init__(self):
        self.face_detector_debug = True


CONFIG = PongConfig()
