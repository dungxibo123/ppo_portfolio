# =============================
# ppo.py (single-file runnable)
# =============================
import os
import argparse
import logging
from datetime import datetime

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import yfinance as yf

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from data import build_dataset
from policy import AttentionFeatureExtractor


# =========================
# Logger
# =========================
def setup_logger(run_name):
    os.makedirs("outputs/attentions/", exist_ok=True)
    log_file = os.path.join("outputs/attentions", f"{run_name}.log")

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger




# =========================
# Environment
# =========================
def make_env(prices, window_size, initial_cash):
    def _init():
        return PortfolioEnv(prices, window_size, initial_cash)
    return _init


def create_vector_env(prices, n_envs, window_size, initial_cash):
    return SubprocVecEnv([
        make_env(prices, window_size, initial_cash)
        for _ in range(n_envs)
    ])

class PortfolioEnv(gym.Env):
    def __init__(self, prices, window_size=60, initial_cash=100000):
        super().__init__()

        self.prices = prices
        self.returns = np.log(prices[1:] / prices[:-1])

        self.window_size = window_size
        self.n_assets = prices.shape[1]
        self.initial_cash = initial_cash

        self.action_space = spaces.Box(low=-5, high=5, shape=(self.n_assets,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_assets, window_size),
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

        # transaction cost
        cost = 0.001 * np.sum(np.abs(weights - self.weights))
        portfolio_return -= cost

        self.portfolio_value *= np.exp(portfolio_return)

        reward = self._differential_sharpe(portfolio_return)

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

    def _differential_sharpe(self, R_t, eta=1/252):
        delta_A = R_t - self.A
        delta_B = R_t**2 - self.B

        numerator = self.B * delta_A - 0.5 * self.A * delta_B
        denominator = (self.B - self.A**2 + 1e-8) ** 1.5

        D_t = numerator / (denominator + 1e-8)

        self.A += eta * delta_A
        self.B += eta * delta_B

        return D_t
    
def train_ppo(prices, args):
    env = create_vector_env(prices, args.n_envs, args.window_size, args.initial_cash)

    policy_kwargs = dict(
        features_extractor_class=AttentionFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            n_heads=4
        ),
        net_arch=[dict(pi=[args.hidden_size], vf=[args.hidden_size])]
    )

    model = PPO(
        "AttentionPolicy",
        env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        learning_rate=args.learning_rate,
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=args.total_timesteps)
    return model

def evaluate(model, prices, args):
    env = PortfolioEnv(prices, args.window_size, args.initial_cash)
    state, _ = env.reset()

    done = False
    values = [env.portfolio_value]

    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        values.append(env.portfolio_value)

    return values

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tickers", nargs='+', default=["XLK"])
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default="2024-01-01")

    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--initial_cash", type=float, default=10000)

    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--n_epochs", type=int, default=10)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--total_timesteps", type=int, default=200000)

    return parser.parse_args()

def main():
    args = parse_args()

    run_name = f"ppo_{'_'.join(args.tickers)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logger(run_name)

    prices_np, _, _, _ = build_dataset(
        tickers=args.tickers,
        start="2010-01-01",
        end="2024-01-01",
    )
    split = int(0.8 * len(prices_np))

    train_prices = prices_np[:split]
    test_prices  = prices_np[split:]
    logger.info(f"Loaded prices: {prices_np.shape}")

    model = train_ppo(train_prices,args)

# Evaluate
    values = evaluate(model, test_prices, args)


    model = train_ppo(train_prices, args)
    values = evaluate(model, test_prices, args)

    final_value = values[-1]

    logger.info(f"Final Portfolio Value: {final_value}")
    print("Final Portfolio Value:", final_value)


if __name__ == "__main__":
    main()
