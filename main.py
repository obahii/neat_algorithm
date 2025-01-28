#!/usr/bin/env python

import pygame as pg
from pong import Game
import neat
import os
import time
import pickle
import json


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


def genome_to_json(genome):
    """Convert a genome into a JSON-serializable dictionary."""
    genome_dict = {
        "key": genome.key,
        "connections": [],
        "nodes": [],
        "fitness": genome.fitness,
    }

    # Save connection genes
    for key, conn in genome.connections.items():
        genome_dict["connections"].append(
            {"key": (key[0], key[1]), "weight": conn.weight, "enabled": conn.enabled}
        )

    # Save node genes
    for key, node in genome.nodes.items():
        genome_dict["nodes"].append(
            {
                "key": key,
                "bias": node.bias,
                "response": node.response,
                "activation": node.activation,
                "aggregation": node.aggregation,
            }
        )

    return genome_dict


def json_to_genome(json_data, config):
    """Convert a JSON dictionary back into a genome."""
    genome = neat.DefaultGenome(0)
    genome.configure_new(config.genome_config)

    # Create node genes
    genome.nodes = {}
    for node_data in json_data["nodes"]:
        node = neat.genes.DefaultNodeGene(node_data["key"])
        node.bias = node_data["bias"]
        node.response = node_data["response"]
        node.activation = node_data["activation"]
        node.aggregation = node_data["aggregation"]
        genome.nodes[node_data["key"]] = node

    genome.connections = {}
    for conn_data in json_data["connections"]:
        conn = neat.genes.DefaultConnectionGene(tuple(conn_data["key"]))
        conn.weight = conn_data["weight"]
        conn.enabled = conn_data["enabled"]
        genome.connections[tuple(conn_data["key"])] = conn

    genome.key = json_data["key"]
    genome.fitness = json_data["fitness"]

    return genome

def run_neat(config):
    checkpoint_dir = "neat_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # population = neat.Checkpointer.restore_checkpoint("neat-checkpoint-5")
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(
        neat.Checkpointer(
            generation_interval=1, filename_prefix=f"{checkpoint_dir}/neat-checkpoint-"
        )
    )
    # population.add_reporter(neat.Checkpointer(1))

    winner = population.run(eval_genomes, 15)
    # with open("model.pkl", "wb") as f:
    #     pickle.dump(winner, f)
    winner_json = genome_to_json(winner)
    with open("model.json", "w") as f:
        json.dump(winner_json, f, indent=2)


def test_ai(config):
    width, height = 1080, 720
    window = pg.display.set_mode((width, height))

    # with open("model.pkl", "rb") as f:
    #     winner = pickle.load(f)
    with open("model.json", "r") as f:
        genome_data = json.load(f)
    winner = json_to_genome(genome_data, config)

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
