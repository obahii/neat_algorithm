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

    def test_ai(self, genome, config):
        nn = neat.nn.FeedForwardNetwork.create(genome, config)

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

            # print(game_info.left_score, game_info.right_score)
            output = nn.activate((self.ball.x, self.ball.y, self.right_paddle.y))
            desicion = output.index(max(output))
            if desicion == 0:
                pass
            elif desicion == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()
            self.game.draw()
            pg.display.update()

        pg.quit()

    def train_ai(self, genome_1, genome_2, config):
        nn_1 = neat.nn.FeedForwardNetwork.create(genome_1, config)
        nn_2 = neat.nn.FeedForwardNetwork.create(genome_2, config)
        run = True
        while run:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    quit()
            nn1_output = nn_1.activate((self.ball.x, self.ball.y, self.left_paddle.y))
            decision_1 = nn1_output.index(max(nn1_output))
            if decision_1 == 0:
                pass
            elif decision_1 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)

            nn2_output = nn_2.activate((self.ball.x, self.ball.y, self.right_paddle.y))
            desicion_2 = nn2_output.index(max(nn2_output))
            if desicion_2 == 0:
                pass
            elif desicion_2 == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)
            game_info = self.game.loop()
            # self.game.draw(draw_score=False, draw_hits=True)
            # pg.display.update()

            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50 or game_info.right_hits > 50:
                self.calculate_fitness(genome_1, genome_2, game_info)

                run = False
                break

    def calculate_fitness(self, genome_1, genome_2, game_info):
        genome_1.fitness += game_info.left_score
        genome_2.fitness += game_info.right_score
        return genome_1, genome_2


# window_width, window_height = 1080, 720
# window = pg.display.set_mode((window_width, window_height))
# pg.display.set_caption("Pong")
# game = PonGame(window, window_width, window_height)
# game.test_ai()

def eval_genomes(genomes, config):
    width, height = 1080, 720
    window = pg.display.set_mode((width, height))
    for i, (genome_id_1, genome_1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome_1.fitness = 0
        for genome_id_2, genome_2 in genomes[i + 1 : ]:
            genome_2.fitness = 0 if genome_2.fitness is None else genome_2.fitness
            game = PonGame(window, width, height)
            game.train_ai(genome_1, genome_2, config)
            
            
            
        
    
    # for genome_id, genome in genomes:
    #     net = neat.nn.FeedForwardNetwork.create(genome, config)
    #     game = PonGame(window, window_width, window_height)
    #     run = True
    #     clock = pg.time.Clock()
    #     while run:
    #         clock.tick(60)
    #         for event in pg.event.get():
    #             if event.type == pg.QUIT:
    #                 run = False
    #                 break
    #         output = net.activate((game.ball.x, game.ball.y, game.ball.x_speed, game.ball.y_speed, game.left_paddle.y, game.right_paddle.y))
    #         if output[0] > 0.5:
    #             game.game.move_paddle(left=True, up=True)
    #         elif output[1] > 0.5:
    #             game.game.move_paddle(left=True, up=False)
    #         game_info = game.game.loop()
    #         genome.fitness = game_info.left_score
    #         if game_info.left_score == 5:
    #             break
    #         game.game.draw()
    #         pg.display.update()
    #     pg.quit()


def run_neat(config):
    population = neat.Checkpointer.restore_checkpoint("neat-checkpoint-23")
    # population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(1))
    
    winner = population.run(eval_genomes, 50)
    with open("model.pkl", "wb") as f:
        pickle.dump(winner, f)


def test_ai(config):
    width, height = 1080, 720
    window = pg.display.set_mode((width, height))
    
    with open("model.pkl", "rb") as f:
        winner = pickle.load(f)
    game = PonGame(window, width, height)
    game.test_ai(winner, config)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    # run_neat(config)
    test_ai(config)
