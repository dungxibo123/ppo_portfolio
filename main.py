import os
import json
import argparse
import logging
from datetime import datetime

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from data import build_dataset
from policy import LSTMFeatureExtractor, AttentionFeatureExtractor


# =========================
# Logger
# =========================
def setup_logger(run_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"{run_name}.log")

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(fh)
    return logger


# =========================
# Callback
# =========================
class LossLoggerCallback(BaseCallback):
    def __init__(self, log_freq):
        super().__init__()
        self.log_freq = log_freq
        self.steps = []
        self.policy_loss = []
        self.value_loss = []
        self.kl = []

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            logs = self.model.logger.name_to_value

            self.steps.append(self.num_timesteps)
            self.policy_loss.append(logs.get("train/policy_gradient_loss", 0))
            self.value_loss.append(logs.get("train/value_loss", 0))
            self.kl.append(logs.get("train/approx_kl", 0))

        return True


# =========================
# Environment
# =========================
class PortfolioEnv(gym.Env):
    def __init__(self, prices, args):
        super().__init__()

        self.prices = prices
        self.returns = np.log(prices[1:] / prices[:-1])

        self.window_size = args.window_size
        self.n_assets = prices.shape[1]
        self.initial_cash = args.initial_cash

        self.transaction_cost = args.transaction_cost
        self.dsr_eta = args.dsr_eta

        self.action_space = spaces.Box(
            low=args.action_low,
            high=args.action_high,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_assets, self.window_size),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.t = self.window_size
        self.portfolio_value = self.initial_cash
        self.weights = np.ones(self.n_assets) / self.n_assets

        self.A = 0.0
        self.B = 0.0

        return self._get_state(), {}

    def step(self, action):
        weights = self._softmax(action)
        r_t = self.returns[self.t]

        portfolio_return = np.dot(weights, r_t)

        turnover = np.sum(np.abs(weights - self.weights))
        cost = self.transaction_cost * turnover
        portfolio_return -= cost

        self.portfolio_value *= np.exp(portfolio_return)

        reward = self._differential_sharpe(portfolio_return)
        reward = np.clip(reward, -10.0, 10.0)

        self.t += 1
        terminated = self.t >= len(self.returns) - 1
        truncated = False

        self.weights = weights
        return self._get_state(), reward, terminated, truncated, {}

    def _get_state(self):
        window = self.returns[self.t - self.window_size:self.t]
        return window.T.astype(np.float32)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / (np.sum(e_x) + 1e-8)

    def _differential_sharpe(self, R_t):
        eta = self.dsr_eta

        delta_A = R_t - self.A
        delta_B = R_t**2 - self.B

        numerator = self.B * delta_A - 0.5 * self.A * delta_B
        denominator = max(self.B - self.A**2, 1e-6) ** 1.5

        D_t = numerator / denominator

        self.A += eta * delta_A
        self.B += eta * delta_B

        return D_t


def make_env(prices, args):
    def _init():
        return PortfolioEnv(prices, args)
    return _init


def create_env(prices, args):
    return SubprocVecEnv([make_env(prices, args) for _ in range(args.n_envs)])


# =========================
# Model
# =========================
def build_model(env, args):
    if args.model_type == "linear":
        policy_kwargs = dict(net_arch=[args.hidden_size, args.hidden_size])

    elif args.model_type == "lstm":
        policy_kwargs = dict(
            features_extractor_class=LSTMFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=args.hidden_size),
            net_arch=[dict(pi=[args.hidden_size], vf=[args.hidden_size])]
        )

    elif args.model_type == "attention":
        policy_kwargs = dict(
            features_extractor_class=AttentionFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=args.attention_dim,
                n_heads=args.n_heads
            ),
            net_arch=[dict(pi=[args.hidden_size], vf=[args.hidden_size])]
        )

    else:
        raise ValueError("Invalid model type")

    return PPO(
        "MlpPolicy",
        env,
        verbose=args.verbose,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs
    )


# =========================
# Train / Evaluate
# =========================
def train(prices, args):
    env = create_env(prices, args)
    model = build_model(env, args)

    callback = LossLoggerCallback(args.steps_per_log) if args.verbose else None

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback
    )

    return model, callback


def evaluate(model, prices, args):
    env = PortfolioEnv(prices, args)
    state, _ = env.reset()

    values = [env.portfolio_value]
    rewards = []

    done = False

    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        values.append(env.portfolio_value)
        rewards.append(reward)

    return values, rewards


# =========================
# Plotting
# =========================
def plot_training(callback, args):
    if callback is None:
        return

    os.makedirs(args.output_dir, exist_ok=True)

    plt.figure()
    plt.plot(callback.steps, callback.policy_loss, label="Policy Loss")
    plt.plot(callback.steps, callback.value_loss, label="Value Loss")
    plt.plot(callback.steps, callback.kl, label="KL")
    plt.legend()
    plt.title("Training Metrics")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(args.output_dir, "training_curve.png"))
    plt.close()


def plot_evaluation(values, args):
    os.makedirs(args.output_dir, exist_ok=True)

    plt.figure()
    plt.plot(values)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.savefig(os.path.join(args.output_dir, "evaluation_curve.png"))
    plt.close()

def save_results(values, rewards, args):
    os.makedirs(args.output_dir, exist_ok=True)

    file_path = os.path.join(args.output_dir, "results.json")

    final_value = float(values[-1])
    avg_dsr = float(np.mean(rewards))

    new_result = {
        "final_portfolio_value": final_value,
        "test_differential_sharpe": avg_dsr,
        "args": vars(args)
    }

    # -------------------------
    # Load existing results
    # -------------------------
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Ensure it's a list
            if not isinstance(data, list):
                data = [data]

        except (json.JSONDecodeError, ValueError):
            # corrupted or empty file
            data = []
    else:
        data = []

    # -------------------------
    # Append new result
    # -------------------------
    data.append(new_result)

    # -------------------------
    # Save back
    # -------------------------
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    return new_result

# =========================
# Args
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, default="linear",
                        choices=["linear", "lstm", "attention"])

    parser.add_argument("--tickers", nargs="+", default=["XLK", "XLV", "XLF", "XLE"])
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2024-01-01")

    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--initial_cash", type=float, default=10000)
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument("--dsr_eta", type=float, default=1/252)
    parser.add_argument("--action_low", type=float, default=-5.0)
    parser.add_argument("--action_high", type=float, default=5.0)

    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_epochs", type=int, default=10)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--test_split", type=float, default=0.8)

    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)

    parser.add_argument("--total_timesteps", type=int, default=200000)

    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--steps_per_log", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="outputs/")

    return parser.parse_args()


# =========================
# Main
# =========================
def main():
    args = parse_args()

    run_name = f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logger(run_name, args.output_dir)

    prices_np, _, _, _ = build_dataset(
        tickers=args.tickers,
        start=args.start,
        end=args.end
    )

    split = int(args.test_split * len(prices_np))
    train_prices = prices_np[:split]
    test_prices = prices_np[split:]

    logger.info(f"Train: {train_prices.shape}, Test: {test_prices.shape}")

    model, callback = train(train_prices, args)
    values, rewards = evaluate(model, test_prices, args)

    if args.verbose:
        plot_training(callback, args)
        plot_evaluation(values, args)

    result = save_results(values, rewards, args)

    logger.info(result)
    print(result)


if __name__ == "__main__":
    main()
