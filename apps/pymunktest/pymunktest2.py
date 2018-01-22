import pymunk

space = pymunk.Space()
space.gravity = (0.0, -900.0)
ballBody = pymunk.Body(10)
ballBody.position = (20, 20)
shape = pymunk.Circle(ballBody, 20, (0, 0))
shape.elasticity = 0
shape.friction = 0
space.add(ballBody, shape)

for i in range(1, 200):
    space.step(0.0002)
    print(ballBody.position)
