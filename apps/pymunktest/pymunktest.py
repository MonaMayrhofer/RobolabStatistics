import pymunk


### Physics stuff
pymunkSpace = pymunk.Space()
pymunkSpace.gravity = (0.0, -9.81)

ballBody = pymunk.Body(10, 25)
ballShape = pymunk.Circle(ballBody, 20, (0, 0))

ballBody.position = 200, 400
pymunkSpace.add(ballBody, ballShape)

while True:
    print(ballBody.position)

    pymunkSpace.step(0.0001)
