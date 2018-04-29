import pygame
import pygame.gfxdraw
import numpy as np
import os
import cv2
from apps.facepong2.pongConfig import CONFIG
from enum import Enum



class TextAlign(Enum):
    LEFT = 0
    RIGHT = 1
    CENTER = 2


class BlitLater:
    def __init__(self, canvas: pygame.Surface, to_blit: pygame.Surface, pos):
        self.canvas = canvas
        self.to_blit = to_blit
        self.pos = pos

    def blit(self):
        self.canvas.blit(self.to_blit, self.pos)

class PongRenderer:
    def __init__(self, cam_size, window_size):
        pygame.init()
        pygame.display.set_caption("Facepong")
        self.display = pygame.display.set_mode(window_size, pygame.FULLSCREEN if CONFIG.graphics.fullscreen else 0)
        self.camSize = cam_size
        self.windowSize = window_size
        self.screen = pygame.Surface(cam_size)
        self.insets = CONFIG.graphics.camera_insets
        self.screenStart = np.subtract(np.divide(self.windowSize, 2), np.divide(self.camSize, 2))

        self.to_blit = []

    def __crop_image(self, image):
        assert self.insets[0] >= 0 and self.insets[1] >= 0
        if self.insets[0] != 0 and self.insets[1] != 0:
            return image[self.insets[0]:-self.insets[0], self.insets[1]:-self.insets[1]]
        if self.insets[1] != 0:
            return image[:, self.insets[1]:-self.insets[1]]
        if self.insets[0] != 0:
            return image[self.insets[0]:-self.insets[0], :]
        return image

    def render(self, video, game, state):
        video = self.__crop_image(video)
        self.screen.fill([0, 0, 0])
        w, h = pygame.display.get_surface().get_size()

        frame = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = np.flip(frame, 0)

        self.display.fill((0, 0, 0))
        state.render(self, frame, game)

        # pygame.draw.rect(self.display, (0, 0, 0), (0, 0, self.display.get_width(), self.display.get_height()))
        self.display.blit(self.screen, self.screenStart)
        self.execute_blits()

        pygame.display.update()

    def circle(self, color, center, radius, width=0):
        pygame.gfxdraw.aacircle(self.screen, center[0], center[1], radius, color)
        pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], radius, color)
        # pygame.draw.circle(self.screen, color, center, radius, width)

    def line(self, color, start, end, width=1):
        pygame.draw.line(self.screen, color, start, end, width)

    def rect(self, color, x, y, width, height, linewidth=0, out=False):
        if y < 0:
            y = y + self.screenStart[1]

        if x < 0:
            x = x + self.screenStart[0]

        if out:
            surf = self.display
        else:
            surf = self.screen
        pygame.draw.rect(surf, color, (x, y, width, height), linewidth)

    def text(self, pos, color, msg, family='Iceland', weight='Regular', size=30, out=False,
             align=(TextAlign.CENTER, TextAlign.CENTER), draw_in_front=True):
        pygame.font.init()

        if os.path.isfile(os.path.join("res", family, family + "-" + weight + ".ttf")):
            myfont = pygame.font.Font("res/Iceland/Iceland-Regular.ttf", size)
        else:
            myfont = pygame.font.SysFont(family, size)

        text_surface = myfont.render(msg, True, color)

        canvas = self.display if out else self.screen

        # Centring
        if pos is None:
            pos = (None, None)
        pos = (pos[0] if pos[0] is not None else self.screen.get_width() / 2,
               pos[1] if pos[1] is not None else self.screen.get_height() / 2)

        # pos = (pos[0] if pos[0] >= 0 else pos[0] + self.screenStart[0],
        #      pos[1] if pos[1] >= 0 else pos[1] + self.screenStart[1])

        if not isinstance(align, tuple):
            align = (align, align)

        # Aligning
        pos = (
            pos[0] if align[0] == TextAlign.LEFT else (
                pos[0] - text_surface.get_width() / 2 if align[0] == TextAlign.CENTER else
                pos[0] - text_surface.get_width()),

            pos[1] if align[1] == TextAlign.LEFT else (
                pos[1] - text_surface.get_height() / 2 if align[1] == TextAlign.CENTER else
                pos[1] - text_surface.get_height()),
        )

        pos = (
            pos[0] + self.screenStart[0] if out else pos[0],
            pos[1] + self.screenStart[1] if out else pos[1],
        )

        # Alpha
        if len(color) > 3:
            tempSurface = pygame.Surface(text_surface.get_size(), pygame.SRCALPHA)
            tempSurface.fill((255, 255, 255, color[3]))
            tempSurface.blit(text_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            text_surface = tempSurface

        if draw_in_front:
            self.blit_later(canvas, text_surface, pos)
        else:
            canvas.blit(text_surface, pos)

    def draw_background(self, img):
        frame = pygame.surfarray.make_surface(img)
        self.screen.blit(frame, (0, 0))

    def blit_later(self, onto, to_blit, pos):
        self.to_blit.append(BlitLater(onto, to_blit, pos))

    def execute_blits(self):
        for b in self.to_blit:
            b.blit()
        self.to_blit.clear()
