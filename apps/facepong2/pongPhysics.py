import pymunk
import abc
from abc import ABCMeta, abstractmethod


class PhysicsObject(metaclass=ABCMeta):
    def add_to(self, space: pymunk.Space):
        space.add(self.body, self.shape)
        return self

    def set_pos(self, pos):
        self.body.position = pos

    def get_pos(self):
        return self.body.position

    def push_pos(self, pos, dt):
        pass


class Ball(PhysicsObject):
    def __init__(self):
        mass = 10
        radius = 25
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        ball_body = pymunk.Body(mass, inertia)
        ball_shape = pymunk.Circle(ball_body, radius, (0, 0))
        ball_shape.elasticity = 0.95
        ball_shape.friction = 0.9
        self.body = ball_body
        self.shape = ball_shape


class Face(PhysicsObject):
    def __init__(self):
        face_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        face_shape = pymunk.Circle(face_body, 50, (0, 0))
        face_shape.elasticity = 0.8
        self.body = face_body
        self.shape = face_shape


class Borders(PhysicsObject):
    def __init__(self, width, height):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        l = 0
        r = width
        t = 0
        b = height
        shape = pymunk.Poly(body, [
            (l - 20, t - 20),
            (l - 20, b + 20),
            (r + 20, b + 20),
            (r + 20, t - 20),
            (l, t - 20),
            (l, t),
            (r, t),
            (r, b),
            (l, b),
            (l, t),
            (l, t - 20),
        ])
        shape.elasticity = 0.8
        self.body = body
        self.shape = shape


class PongPhysics:
    def __init__(self, width, height):
        print()
        insets = (80, 20)  # Top, Bottom

        pymunkSpace = pymunk.Space()
        pymunkSpace.gravity = (0.0, 0.0)
        self.space = pymunkSpace
        self.ball = Ball().add_to(pymunkSpace)
        self.faceOne = Face().add_to(pymunkSpace)
        self.faceTwo = Face().add_to(pymunkSpace)

        self.faceOne.set_pos((0, 0))
        self.faceTwo.set_pos((0, 0))

        Borders(width, height).add_to(pymunkSpace)

    def tick(self, delta):
        self.space.step(delta)
