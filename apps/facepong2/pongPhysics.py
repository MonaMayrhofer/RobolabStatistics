import pymunk
import abc
from abc import ABCMeta, abstractmethod
import numpy as np
import random
from apps.facepong2.pongConfig import CONFIG

class PhysicsObject(metaclass=ABCMeta):
    def __init__(self):
        self.was_none = True
        self.body = None  # Ducktype
        self.shape = None  # Ducktype

    def add_to(self, space: pymunk.Space):
        space.add(self.body, self.shape)
        return self

    def set_pos(self, pos):
        self.was_none = False
        self.body.position = pos

    def get_pos(self):
        return self.body.position

    def nonify(self):
        self.was_none = True

    def is_none(self):
        return self.was_none

    def push_pos(self, pos, dt):
        if pos is None:
            self.nonify()
            return
        if self.was_none:
            self.set_pos(pos)
        self.was_none = False
        self.body.velocity = np.divide(np.subtract(pos, self.get_pos()), max(dt, 0.00001))


class Ball(PhysicsObject):
    def __init__(self):
        super().__init__()
        mass = 10
        self.radius = 25
        inertia = pymunk.moment_for_circle(mass, 0, self.radius, (0, 0))
        ball_body = pymunk.Body(mass, inertia)
        ball_shape = pymunk.Circle(ball_body, self.radius, (0, 0))
        ball_shape.elasticity = 0.95
        ball_shape.friction = 0.9
        self.body = ball_body
        self.shape = ball_shape

    def throw_in(self, pos):
        self.set_pos(pos)
        if random.randint(0, 1) == 0:
            self.body.velocity = (50, 0)
        else:
            self.body.velocity = (-50, 0)

    def normalize_speed(self):
        self.body.velocity = self.resize(self.body.velocity, CONFIG.max_ball_speed)

    @staticmethod
    def resize(l_tuple, l_new_len):
        length = (l_tuple[0] ** 2 + l_tuple[1] ** 2) ** 0.5
        if length > l_new_len:
            normal = (l_tuple[0] / length * l_new_len, l_tuple[1] / length * l_new_len)
        else:
            normal = l_tuple
        return normal


class Face(PhysicsObject):
    def __init__(self):
        super().__init__()
        self.radius = 50
        face_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        face_shape = pymunk.Circle(face_body, self.radius, (0, 0))
        face_shape.elasticity = 0.8
        self.body = face_body
        self.shape = face_shape


class Borders(PhysicsObject):
    def __init__(self, width, height):
        super().__init__()
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        middle = 50
        left = 0
        lo = left - middle
        right = width
        ro = right + middle
        top = 0
        to = top - middle
        bottom = height
        bo = bottom + middle

        shape = [
            pymunk.Poly(body, [
                (lo, to),
                (lo, top),
                (ro, top),
                (ro, to)
            ]),
            pymunk.Poly(body, [
                (lo, bottom),
                (lo, bo),
                (ro, bo),
                (ro, bottom)
            ])
        ]
        for s in shape:
            s.elasticity = 0.8
        self.body = body
        self.shape = shape

    def add_to(self, space: pymunk.Space):
        space.add(self.body)
        for shape in self.shape:
            space.add(shape)
        return self


class PongPhysics:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # CREATE WORLD
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.ball = Ball().add_to(self.space)
        self.faceOne = Face().add_to(self.space)
        self.faceTwo = Face().add_to(self.space)
        Borders(self.width, self.height).add_to(self.space)

    def tick(self, delta):
        self.space.step(delta)
        self.ball.normalize_speed()

    def is_invalid_state(self):
        pos = self.ball.get_pos()
        return pos[1] < 0 or pos[1] > self.height

    def throw_in(self):
        print("Throw In")
        self.ball.throw_in((self.width / 2, self.height / 2))

    def get_win(self):
        pos = self.ball.get_pos()
        if pos[0] < 0:
            return 1
        if pos[0] > self.width:
            return -1
        return 0
