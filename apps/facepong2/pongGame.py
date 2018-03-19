from apps.facepong2 import camOpener
from apps.facepong2.pongFaceDetector import PongFaceDetector
import cv2
import numpy as np

class PongGame:
    def __init__(self, file='FrontalFace.xml'):
        self.cap = camOpener.open_cam()
        _, img = self.cap.read()
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.faceDetector = PongFaceDetector(file)
        self.paused = False
        self.winPaused = False


    def run(self):
        try:
            while True:
                print("LOop")
                # == Calc FPS

                # == Read Image ==
                _, img = self.cap.read()
                debug = np.zeros(img.shape)
                cv2.flip(img, 1, img)

                field_size = int(img.shape[1] / 3)

                rightface, leftface = self.faceDetector.get_faces(img, (30, 30), (300, 300), field_size, field_size*2)

                # == Game Loop ==
                if not self.paused and not self.winPaused:

                    x1, y1, w1, h1 = rightface[0]
                    x2, y2, w2, h2 = leftface[1]

                    currFaces = [(x1 + w1 / 2, y1 + h1 / 2), (x2 + w2 / 2, y2 + h2 / 2)]
                    faceVelocities = np.divide(np.subtract(currFaces, lastFaces), max(delta, 0.00001))
                    lastFaces = currFaces

                    faceOneBody.velocity = faceVelocities[0] * slowdown
                    faceTwoBody.velocity = faceVelocities[1] * slowdown

                    ballBody.velocity = resize(ballBody.velocity, speed)

                    if delta != 0:
                        pymunkSpace.step(delta / slowdown)

                    # Move ball
                    ballPos = ballBody.position

                    # Detect goals
                    if ballPos[0] < 25:
                        # RESET
                        pointsRight += 1
                        reset()
                    elif ballPos[0] + 25 > width:
                        # RESET
                        pointsLeft += 1
                        reset()

                    if ballPos[0] < -borderThickness or ballPos[1] < -borderThickness or ballPos[
                        0] > width + borderThickness or \
                            ballPos[1] > height + borderThickness:
                        reset()

                    # Speed increase
                    if speed < max_speed():
                        speed = speed * 1.001

                # == Detect win ==
                if winTime == 0 and (pointsLeft == pointsToWin or pointsRight == pointsToWin):
                    winTime = time.time()
                    winPaused = True
                    win()

                if pointsLeft == pointsToWin:
                    cv2.putText(img, "Spieler links gewinnt!", (int(width / 2) - 200, int(height / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                elif pointsRight == pointsToWin:
                    cv2.putText(img, "Spieler rechts gewinnt!", (int(width / 2) - 200, int(height / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)

                # == Reset on win ==
                if winTime != 0 and time.time() - winTime > 3:
                    winPaused = False
                    pointsLeft = 0
                    pointsRight = 0
                    winTime = 0

                # == Crop Image ==
                img = img[insets[0]:-insets[0], insets[1]:-insets[1]]

                # == Debug Data ==

                # == Show Points

                screen.fill([0, 0, 0])
                w, h = pygame.display.get_surface().get_size()

                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame = np.rot90(frame)
                frame = np.flip(frame, 0)
                frame = pygame.surfarray.make_surface(frame)
                screen.blit(frame, (0, 0))

                # == Draw Ball ==
                pygame.draw.circle(screen, (255, 0, 0), (int(ballPos[0]), int(ballPos[1])), 50)
                pygame.draw.line(screen, (255, 0, 0), (int(img.shape[1] / 3), 0), (int(img.shape[1] / 3), img.shape[0]))
                pygame.draw.line(screen, (255, 0, 0), (int(img.shape[1] / 3 * 2), 0),
                                 (int(img.shape[1] / 3 * 2), img.shape[0]))

                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == 27 or event.key == 113:
                            sys.exit(0)
                    elif event.type == pygame.VIDEORESIZE:
                        screen = pygame.display.set_mode(event.dict['size'],
                                                         pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.FULLSCREEN)
                        pygame.display.flip()

                pygame.display.update()
        except KeyboardInterrupt:
            pygame.quit()
        except SystemExit:
            pygame.quit()