import pymunk
import abc
from abc import ABCMeta, abstractmethod
import numpy as np
import random


class PhysicsObject(metaclass=ABCMeta):
    def __init__(self):
        self.was_none = True

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
        m = 50
        l = 0
        lo = l - m
        r = width
        ro = r + m
        t = 0
        to = t - m
        b = height
        bo = b + m

        """
        #LeftBody
        
            pymunk.Poly(body, [
                (lo, to),
                (lo, bo),
                (l, bo),
                (l, to)
            ]),
            
        #RightBody
        
            pymunk.Poly(body, [
                (r, to),
                (r, bo),
                (ro, bo),
                (ro, to)
            ]),
        
        """

        shape = [
            pymunk.Poly(body, [
                (lo, to),
                (lo, t),
                (ro, t),
                (ro, to)
            ]),
            pymunk.Poly(body, [
                (lo, b),
                (lo, bo),
                (ro, bo),
                (ro, b)
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

        # SETUP PHYSICS
        self.throw_in()

    def tick(self, delta):
        self.space.step(delta)

    def is_invalid_state(self):
        pos = self.ball.get_pos()
        return pos[1] < 0 or pos[1] > self.height

    def throw_in(self):
        print("Throw In")
        self.ball.throw_in((self.width/2, self.height/2))

    def get_win(self):
        pos = self.ball.get_pos()
        if pos[0] < 0:
            return 1
        if pos[0] > self.width:
            return -1
        return 0

