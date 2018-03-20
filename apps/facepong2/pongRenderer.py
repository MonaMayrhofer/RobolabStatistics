import pygame
import numpy as np
import cv2


class PongRenderer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("OpenCV camera stream on Pygame")
        self.screen = pygame.display.set_mode([1280, 720], pygame.FULLSCREEN & 0)
        self.insets = (0, 0)

    def __crop_image(self, image):
        assert self.insets[0] >= 0 and self.insets[1] >= 0
        if self.insets[0] != 0 and self.insets[1] != 0:
            return image[self.insets[0]:-self.insets[0], self.insets[1]:-self.insets[1]]
        if self.insets[1] != 0:
            return image[:, self.insets[1]:-self.insets[1]]
        if self.insets[0] != 0:
            return image[self.insets[0]:-self.insets[0], :]
        return image

    def render(self, video, physics):
        video = self.__crop_image(video)
        self.screen.fill([0, 0, 0])
        w, h = pygame.display.get_surface().get_size()

        frame = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = np.flip(frame, 0)
        frame = pygame.surfarray.make_surface(frame)
        self.screen.blit(frame, (0, 0))

        ballPos = physics.ball.get_pos()
        faceAPos = physics.faceOne.get_pos()
        faceBPos = physics.faceTwo.get_pos()
        # == Circles ==
        pygame.draw.circle(self.screen, (255, 0, 0), (int(ballPos[0]), int(ballPos[1])), 50)
        if faceAPos is not None:
            pygame.draw.circle(self.screen, (0, 255, 0), (int(faceAPos[0]), int(faceAPos[1])), 50)
        if faceBPos is not None:
            pygame.draw.circle(self.screen, (0, 0, 255), (int(faceBPos[0]), int(faceBPos[1])), 50)
        pygame.draw.line(self.screen, (255, 0, 0), (int(video.shape[1] / 3), 0), (int(video.shape[1] / 3), video.shape[0]))
        pygame.draw.line(self.screen, (255, 0, 0), (int(video.shape[1] / 3 * 2), 0),
                         (int(video.shape[1] / 3 * 2), video.shape[0]))

        pygame.display.update()