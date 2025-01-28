#!/usr/bin/env python
import pygame as pg
from pong import Game
import neat
import os
import time
import json
import math


class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.ball = self.game.ball
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.previous_ball_pos = (self.ball.x, self.ball.y)
        self.was_ball_reachable = True

    def get_ball_speed(self):
        current_pos = (self.ball.x, self.ball.y)
        dx = current_pos[0] - self.previous_ball_pos[0]
        dy = current_pos[1] - self.previous_ball_pos[1]
        self.previous_ball_pos = current_pos
        return dx, dy

    def predict_ball_intersection(self):
        dx, dy = self.get_ball_speed()
        if dx == 0:
            return self.ball.x, self.ball.y

        # Calculate time to reach paddle's x position
        if dx > 0:  # Ball moving right
            t = (self.right_paddle.x - self.ball.x) / dx
        else:  # Ball moving left
            t = (self.left_paddle.x - self.ball.x) / dx

        future_y = self.ball.y + dy * t
        return future_y

    def get_normalized_inputs(self, paddle):
        # Get ball speed
        ball_dx, ball_dy = self.get_ball_speed()

        # Normalize positions by game dimensions
        norm_ball_x = self.ball.x / self.game.game_width
        norm_ball_y = self.ball.y / self.game.game_height
        norm_paddle_y = paddle.y / self.game.game_height

        # Normalize speeds
        max_speed = 10.0  # Adjust based on your game's maximum speeds
        norm_ball_dx = ball_dx / max_speed
        norm_ball_dy = ball_dy / max_speed

        # Calculate distance to ball
        dx = self.ball.x - paddle.x
        dy = self.ball.y - paddle.y
        distance = math.sqrt(dx * dx + dy * dy)
        norm_distance = distance / math.sqrt(
            self.game.game_width**2 + self.game.game_height**2
        )

        # Predict intersection point
        intersection_y = self.predict_ball_intersection()
        norm_intersection = intersection_y / self.game.game_height

        return [
            norm_ball_x,
            norm_ball_y,
            norm_paddle_y,
            norm_ball_dx,
            norm_ball_dy,
            norm_distance,
            norm_intersection,
        ]

    def test_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)

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

            inputs = self.get_normalized_inputs(self.right_paddle)
            output = net.activate(inputs)
            decision = output.index(max(output))

            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()
            self.game.draw()
            pg.display.update()

        pg.quit()

    def train_ai(self, genome1, genome2, config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

        max_hits = 50
        start_time = time.time()
        max_time = 60  # 1 minute timeout

        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    return

            # Left paddle (Genome 1)
            inputs1 = self.get_normalized_inputs(self.left_paddle)
            output1 = net1.activate(inputs1)
            decision1 = output1.index(max(output1))

            if decision1 == 1:
                self.game.move_paddle(left=True, up=True)
            elif decision1 == 2:
                self.game.move_paddle(left=True, up=False)

            # Right paddle (Genome 2)
            inputs2 = self.get_normalized_inputs(self.right_paddle)
            output2 = net2.activate(inputs2)
            decision2 = output2.index(max(output2))

            if decision2 == 1:
                self.game.move_paddle(left=False, up=True)
            elif decision2 == 2:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()

            # Check termination conditions
            if (
                game_info.left_score >= 1
                or game_info.right_score >= 1
                or game_info.left_hits > max_hits
                or game_info.right_hits > max_hits
                or time.time() - start_time > max_time
            ):
                self.calculate_fitness(genome1, genome2, game_info)
                break

    def calculate_fitness(self, genome1, genome2, game_info):
        # Base points for scoring
        score_weight = 10
        genome1.fitness += game_info.left_score * score_weight
        genome2.fitness += game_info.right_score * score_weight

        # Reward for successful hits
        hit_weight = 0.5
        genome1.fitness += game_info.left_hits * hit_weight
        genome2.fitness += game_info.right_hits * hit_weight

        # Penalty for letting opponent score
        miss_penalty = 2
        if self.was_ball_reachable:
            genome1.fitness -= game_info.right_score * miss_penalty
            genome2.fitness -= game_info.left_score * miss_penalty


def eval_genomes(genomes, config):
    width, height = 1080, 720
    window = pg.display.set_mode((width, height))

    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break

        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i + 1 :]:
            genome2.fitness = 0 if genome2.fitness is None else genome2.fitness
            game = PongGame(window, width, height)
            game.train_ai(genome1, genome2, config)


def run_neat(config):
    # Use this to start fresh
    pop = neat.Population(config)

    # Or use this to continue from a checkpoint
    # pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-X')

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(5))

    winner = pop.run(eval_genomes, 50)

    with open("best_genome.json", "w") as f:
        json.dump(genome_to_json(winner), f)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    run_neat(config)
    # test_ai(config)
