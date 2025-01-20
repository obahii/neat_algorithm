import pygame as pg
import tensorflow as tf
import neat
import os
import pickle
import numpy as np
from pong import Game
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple


class NEATPongNet(tf.keras.Model):
    def __init__(self, genome, config):
        super().__init__()
        self.genome = genome
        self.config = config

        # Convert NEAT genome to TensorFlow layers
        self.model_layers = self._build_layers()

    def _build_layers(self):
        return [
            tf.keras.layers.Dense(10, activation="relu", input_shape=(3,)),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.model_layers:
            x = layer(x)
        return x


class GPUPongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.ball = self.game.ball
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle

        # Check if GPU is available
        self.gpus = tf.config.list_physical_devices("GPU")
        if self.gpus:
            # Enable memory growth to prevent TF from allocating all GPU memory
            for gpu in self.gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

    def get_game_state(self) -> tf.Tensor:
        # Convert game state to TensorFlow tensor
        state = tf.convert_to_tensor(
            [[self.ball.x, self.ball.y, self.right_paddle.y]], dtype=tf.float32
        )
        return state

    def train_ai_batch(self, genomes: List[Tuple], config, batch_size=32):
        """Train multiple AI instances in parallel using GPU batching"""
        networks = [NEATPongNet(genome, config) for genome, _ in genomes]

        # Create game instances for each network pair
        games = [
            GPUPongGame(self.game.window, self.game.window_width, self.game.window_height)
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

    @tf.function
    def _predict_batch(self, network, states):
        """GPU-accelerated batch prediction"""
        return network(states)

    def _process_game_batch(self, games, networks):
        results = []
        max_steps = 2000  # Prevent infinite games

        for step in range(max_steps):
            # Get states for all games
            states = tf.concat([game.get_game_state() for game in games], axis=0)

            # Process all networks in parallel on GPU
            outputs = [self._predict_batch(net, states) for net in networks]

            # Update game states based on network outputs
            for game_idx, game in enumerate(games):
                left_output = outputs[game_idx * 2]
                right_output = outputs[game_idx * 2 + 1]

                # Convert outputs to actions
                left_action = tf.argmax(left_output[game_idx : game_idx + 1], axis=1)[0]
                right_action = tf.argmax(right_output[game_idx : game_idx + 1], axis=1)[
                    0
                ]

                # Apply actions
                self._apply_action(game, left_action.numpy(), is_left=True)
                self._apply_action(game, right_action.numpy(), is_left=False)

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
    window = pg.display.set_mode((width, height), flags=pg.SWSURFACE)
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
    # Fix for OpenGL context / X11 display issues
    os.environ["SDL_VIDEODRIVER"] = "x11"
    os.environ["DISPLAY"] = ":0"

    # Enable mixed precision training for better GPU performance
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Initialize pygame before creating the window
    pg.init()

    # Set pygame OpenGL attributes
    pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
    pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
    pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run_neat_gpu(config_path)
