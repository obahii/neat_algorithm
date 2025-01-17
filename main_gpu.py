import pygame as pg
import torch
import torch.nn as nn
import neat
import os
import pickle
import numpy as np
from pong import Game
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple


class NEATPongNet(nn.Module):
    def __init__(self, genome, config):
        super().__init__()
        self.genome = genome
        self.config = config

        # Convert NEAT genome to PyTorch layers
        self.layers = self._build_layers()
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def _build_layers(self):
        layers = nn.ModuleList()

        # Simplified network architecture - you can make this more complex
        layers.append(nn.Linear(3, 10))  # 3 inputs: ball_x, ball_y, paddle_y
        layers.append(nn.ReLU())
        layers.append(nn.Linear(10, 3))  # 3 outputs: stay, up, down
        layers.append(nn.Softmax(dim=1))

        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GPUPongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.ball = self.game.ball
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_game_state(self) -> torch.Tensor:
        # Convert game state to tensor and move to GPU
        state = torch.tensor(
            [self.ball.x, self.ball.y, self.right_paddle.y], dtype=torch.float32
        ).to(self.device)

        return state.unsqueeze(0)  # Add batch dimension

    def train_ai_batch(self, genomes: List[Tuple], config, batch_size=32):
        """Train multiple AI instances in parallel using GPU batching"""
        networks = [
            NEATPongNet(genome, config).to(self.device) for genome, _ in genomes
        ]

        # Create game instances for each network pair
        games = [
            GPUPongGame(self.game.window, self.game.game_width, self.game.game_height)
            for _ in range(len(networks) // 2)
        ]

        # Process games in batches
        with ThreadPoolExecutor() as executor:
            batch_results = list(
                executor.map(
                    self._process_game_batch,
                    [
                        games[i : i + batch_size]
                        for i in range(0, len(games), batch_size)
                    ],
                    [
                        networks[i : i + batch_size * 2]
                        for i in range(0, len(networks), batch_size * 2)
                    ],
                )
            )

        # Update fitness values
        for game_result in batch_results:
            for genome_idx, fitness in game_result:
                genomes[genome_idx][1].fitness = fitness

    def _process_game_batch(self, games, networks):
        results = []
        max_steps = 2000  # Prevent infinite games

        for step in range(max_steps):
            # Get states for all games
            states = torch.cat([game.get_game_state() for game in games])

            # Process all networks in parallel on GPU
            with torch.no_grad():
                outputs = [net(states) for net in networks]

            # Update game states based on network outputs
            for game_idx, game in enumerate(games):
                left_output = outputs[game_idx * 2]
                right_output = outputs[game_idx * 2 + 1]

                # Convert outputs to actions
                left_action = torch.argmax(left_output).item()
                right_action = torch.argmax(right_output).item()

                # Apply actions
                self._apply_action(game, left_action, is_left=True)
                self._apply_action(game, right_action, is_left=False)

                # Update game state
                game_info = game.game.loop()
                game.game.draw(draw_score=False, draw_hits=True)

                # Check if game is finished
                if (
                    game_info.left_score >= 1
                    or game_info.right_score >= 1
                    or game_info.left_hits > 50
                    or game_info.right_hits > 50
                ):
                    results.append(
                        (game_idx * 2, game_info.left_score + game_info.left_hits * 0.1)
                    )
                    results.append(
                        (
                            game_idx * 2 + 1,
                            game_info.right_score + game_info.right_hits * 0.1,
                        )
                    )

            pg.display.update()

        return results

    @staticmethod
    def _apply_action(game, action, is_left):
        if action == 1:  # Move up
            game.game.move_paddle(left=is_left, up=True)
        elif action == 2:  # Move down
            game.game.move_paddle(left=is_left, up=False)


def run_neat_gpu(config_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Initialize PyGame
    width, height = 1080, 720
    window = pg.display.set_mode((width, height))
    game = GPUPongGame(window, width, height)

    # Custom evaluation function using GPU
    def eval_genomes_gpu(genomes, config):
        game.train_ai_batch(genomes, config)

    # Run evolution
    winner = population.run(eval_genomes_gpu, 50)

    # Save the winner
    with open("model_gpu.pkl", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run_neat_gpu(config_path)
