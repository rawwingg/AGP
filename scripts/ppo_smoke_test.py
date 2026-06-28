"""Quick sanity check that Stable-Baselines3 PPO is installed and training works."""

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym


def main() -> None:
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ppo_smoke/")
    model.learn(total_timesteps=10_000)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
