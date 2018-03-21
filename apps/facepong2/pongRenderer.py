import pygame
import numpy as np
import cv2


class PongRenderer:
    def __init__(self, camSize, windowSize):
        pygame.init()
        pygame.display.set_caption("OpenCV camera stream on Pygame")
        self.display = pygame.display.set_mode(windowSize, pygame.FULLSCREEN & 0)
        self.camSize = camSize
        self.windowSize = windowSize
        self.screen = pygame.Surface(camSize)
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

    def render(self, video, physics, state):
        video = self.__crop_image(video)
        self.screen.fill([0, 0, 0])
        w, h = pygame.display.get_surface().get_size()

        frame = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = np.flip(frame, 0)

        state.render(self, frame, physics)

        self.display.blit(self.screen, np.subtract(np.divide(self.windowSize, 2), np.divide(self.camSize, 2)))

        pygame.display.update()

    def circle(self, color, center, radius, width=0):
        pygame.draw.circle(self.screen, color, center, radius, width)

    def line(self, color, start, end, width=1):
        pygame.draw.line(self.screen, color, start, end, width)

    def text(self, pos, color, msg):
        pygame.font.init()
        myfont = pygame.font.SysFont('Helvetica', 30)
        textsurface = myfont.render(msg, True, color)
        self.screen.blit(textsurface, pos)

    def draw_background(self, img):
        frame = pygame.surfarray.make_surface(img)
        self.screen.blit(frame, (0, 0))
