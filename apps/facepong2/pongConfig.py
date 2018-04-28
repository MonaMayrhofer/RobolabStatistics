class PongConfig:
    def __init__(self):
        self.graphics = PongGraphicsConfig()
        self.fps_interval = 0.5


class PongGraphicsConfig:
    def __init__(self):
        self.face_border_thickness = 10
        self.middle_field_brightness = 0.5


CONFIG = PongConfig()
