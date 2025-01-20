import tensorflow as tf
import numpy as np
import tensorflowjs as tfjs
import os
from tqdm import tqdm
import time


class PongEnv:
    def __init__(self, width=1080, height=720):
        self.width = width
        self.height = height
        self.paddle_width = 20
        self.paddle_height = 100
        self.ball_radius = 15
        self.paddle_speed = 10
        self.ball_speed = 10
        self.reset()

    def reset(self):
        # Reset ball to center
        self.ball_pos = np.array([self.width / 2, self.height / 2], dtype=np.float32)
        self.ball_vel = np.array([self.ball_speed, 0], dtype=np.float32)
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        self.ball_vel = self.ball_speed * np.array([np.cos(angle), np.sin(angle)])

        # Reset paddles
        self.left_paddle = self.height / 2 - self.paddle_height / 2
        self.right_paddle = self.height / 2 - self.paddle_height / 2

        return self._get_state()

    def _get_state(self):
        return np.array(
            [
                self.ball_pos[0] / self.width,  # normalized x position
                self.ball_pos[1] / self.height,  # normalized y position
                self.left_paddle / self.height,  # normalized left paddle position
                self.ball_vel[0] / self.ball_speed,  # normalized x velocity
                self.ball_vel[1] / self.ball_speed,  # normalized y velocity
            ],
            dtype=np.float32,
        )

    def step(self, action):
        # Action: 0 (up), 1 (stay), 2 (down)
        # Update left paddle position
        if action == 0:  # up
            self.left_paddle = max(0, self.left_paddle - self.paddle_speed)
        elif action == 2:  # down
            self.left_paddle = min(
                self.height - self.paddle_height, self.left_paddle + self.paddle_speed
            )

        # Simple AI for right paddle (to generate training data)
        if self.ball_pos[1] > self.right_paddle + self.paddle_height / 2:
            self.right_paddle = min(
                self.height - self.paddle_height, self.right_paddle + self.paddle_speed
            )
        elif self.ball_pos[1] < self.right_paddle + self.paddle_height / 2:
            self.right_paddle = max(0, self.right_paddle - self.paddle_speed)

        # Update ball position
        self.ball_pos += self.ball_vel

        # Ball collision with top and bottom
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.height:
            self.ball_vel[1] *= -1

        # Ball collision with paddles
        if (
            self.ball_pos[0] <= self.paddle_width
            and self.left_paddle
            <= self.ball_pos[1]
            <= self.left_paddle + self.paddle_height
        ):
            self.ball_vel[0] *= -1
            reward = 1.0  # Reward for successful hit
        elif (
            self.ball_pos[0] >= self.width - self.paddle_width
            and self.right_paddle
            <= self.ball_pos[1]
            <= self.right_paddle + self.paddle_height
        ):
            self.ball_vel[0] *= -1
            reward = 0.0
        # Ball out of bounds
        elif self.ball_pos[0] <= 0:
            reward = -1.0  # Penalize missing the ball
            done = True
            return self._get_state(), reward, done
        elif self.ball_pos[0] >= self.width:
            reward = 0.0
            done = True
            return self._get_state(), reward, done
        else:
            reward = 0.0

        done = False
        return self._get_state(), reward, done


def create_model():
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU acceleration enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(128, activation="relu", input_shape=(5,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def collect_experience(env, num_episodes=1000):
    states = []
    actions = []
    rewards = []

    for _ in tqdm(range(num_episodes), desc="Collecting experience"):
        state = env.reset()
        done = False

        while not done:
            # Use heuristic to determine action
            ball_y = state[1]  # normalized ball y position
            paddle_y = state[2]  # normalized paddle y position
            paddle_height_norm = env.paddle_height / env.height

            if ball_y < paddle_y:
                action = 0  # move up
            elif ball_y > paddle_y + paddle_height_norm:
                action = 2  # move down
            else:
                action = 1  # stay

            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int32),
        np.array(rewards, dtype=np.float32),
    )


def train_model(save_path="public/ai-models/pong-model"):
    # Create environment
    env = PongEnv()

    # Collect training data
    print("Collecting training data...")
    states, actions, rewards = collect_experience(env)

    # Create and train model
    print("Creating model...")
    model = create_model()

    # Convert data to TensorFlow datasets
    dataset = (
        tf.data.Dataset.from_tensor_slices((states, actions))
        .shuffle(10000)
        .batch(256)
        .prefetch(tf.data.AUTOTUNE)
    )

    print("Training model...")
    start_time = time.time()

    # Train with early stopping
    history = model.fit(
        dataset,
        epochs=50,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=0.0001
            ),
        ],
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Create output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Save the model in TensorFlow.js format
    print(f"Saving model to {save_path}")
    tfjs.converters.save_keras_model(model, save_path)

    # Test the model
    print("\nTesting model...")
    test_episodes = 10
    total_reward = 0

    for episode in range(test_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = np.argmax(model.predict(state[np.newaxis], verbose=0))
            state, reward, done = env.step(action)
            episode_reward += reward

        total_reward += episode_reward
        print(f"Episode {episode + 1} reward: {episode_reward}")

    print(f"\nAverage test reward: {total_reward / test_episodes}")
    return model, history


if __name__ == "__main__":
    model, history = train_model()
