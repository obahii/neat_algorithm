#!/usr/bin/env python

import pygame as pg
from pong import Game
import neat
import os
import time
import pickle

class PonGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.ball = self.game.ball
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
    
    def test_ai(self):
        run = True
        clock = pg.time.Clock()
        while run:
            clock.tick(60)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    run = False
                    break
            keys = pg.key.get_pressed()
            if keys[pg.K_w]:
                self.game.move_paddle(left=True, up=True)
            elif keys[pg.K_s]:
                self.game.move_paddle(left=True, up=False)
            game_info = self.game.loop()
            # print(game_info.left_score, game_info.right_score)
            self.game.draw()
            pg.display.update()

        pg.quit()
    



window_width, window_height = 1080, 720
window = pg.display.set_mode((window_width, window_height))
pg.display.set_caption("Pong")
game = PonGame(window, window_width, window_height)
game.test_ai()

    
    