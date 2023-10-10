from collections import deque
import copy
import itertools
import random
import shutil
import time
from typing import NamedTuple, Optional
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

import pacbot_rs

from models import QNet
from utils import lerp


wandb.init(
    project="pacbot-dqn",
    config={
        "learning_rate": 0.001,
        "batch_size": 64,
        "num_iters": 150_000,
        "replay_buffer_size": 10_000,
        "target_network_update_steps": 500,  # Update the target network every ___ steps.
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "discount_factor": 0.999,
        "reward_scale": 1 / 50,
        "grad_clip_norm": 0.1,
    },
    # mode="disabled",
)


# Initialize the environment.
gym = pacbot_rs.PacmanGym(random_start=True)
gym.reset()
last_obs = torch.from_numpy(gym.obs_numpy())
obs_shape = last_obs.shape
num_actions = 5
exploration_epsilon = wandb.config.initial_epsilon

# Initialize the Q network.
q_net = QNet(obs_shape, num_actions)
print(f"q_net has {sum(p.numel() for p in q_net.parameters())} parameters")


class ReplayItem(NamedTuple):
    obs: torch.Tensor
    action: int
    reward: int
    next_obs: Optional[torch.Tensor]


replay_buffer: deque[ReplayItem] = deque(maxlen=wandb.config.replay_buffer_size)


@torch.no_grad()
def generate_experience_step():
    global last_obs

    # Choose an action (using q_net and epsilon-greedy for exploration).
    # TODO: invalid action masking?
    if random.random() < exploration_epsilon:
        action = random.randrange(num_actions)
    else:
        q_net.eval()
        action_values = q_net(last_obs.unsqueeze(0)).squeeze(0)
        action = action_values.argmax()

    # Perform the action and observe the transition.
    reward, done = gym.step(action)
    next_obs = None if done else torch.from_numpy(gym.obs_numpy())

    # Add the transition to the replay buffer.
    replay_buffer.append(ReplayItem(last_obs, action, reward, next_obs))

    # Reset the environment if necessary and update last_obs.
    if next_obs is None:
        gym.reset()
        last_obs = torch.from_numpy(gym.obs_numpy())
    else:
        last_obs = next_obs


# Fill the replay buffer with random initial experience.
while len(replay_buffer) < replay_buffer.maxlen:
    generate_experience_step()


@torch.no_grad()
def evaluate_episode(max_steps: int = 1000) -> tuple[int, int]:
    """
    Performs a single evaluation episode.

    Returns (score, total_steps).
    """
    gym = pacbot_rs.PacmanGym(random_start=True)
    gym.reset()

    q_net.eval()

    for step_num in range(1, max_steps + 1):
        obs = torch.from_numpy(gym.obs_numpy())
        action_values = q_net(obs.unsqueeze(0)).squeeze(0)
        _, done = gym.step(action_values.argmax())

        if done:
            break

    return (gym.score(), step_num)


def train():
    global exploration_epsilon

    optimizer = torch.optim.Adam(q_net.parameters(), lr=wandb.config.learning_rate)

    for iter_num in tqdm(range(wandb.config.num_iters), smoothing=0.01):
        if iter_num % wandb.config.target_network_update_steps == 0:
            # Update the target network.
            target_q_net = copy.deepcopy(q_net)
            target_q_net.eval()

        # Sample and collate a batch.
        batch = random.sample(replay_buffer, k=wandb.config.batch_size)
        obs_batch = torch.stack([item.obs for item in batch])
        next_obs_batch = torch.stack(
            [
                torch.zeros(obs_shape) if item.next_obs is None else item.next_obs
                for item in batch
            ]
        )
        done_mask = torch.tensor([item.next_obs is None for item in batch])
        action_batch = torch.tensor([item.action for item in batch])
        reward_batch = torch.tensor(
            [item.reward * wandb.config.reward_scale for item in batch]
        )

        # Get the target Q values.
        with torch.no_grad():
            returns = target_q_net(next_obs_batch).amax(dim=1)
            discounted_returns = wandb.config.discount_factor * returns
            discounted_returns[done_mask] = 0.0
            target_q_values = reward_batch + discounted_returns

        # Compute the loss and update the parameters.
        q_net.train()
        optimizer.zero_grad()

        all_predicted_q_values = q_net(obs_batch)
        predicted_q_values = all_predicted_q_values[range(len(batch)), action_batch]
        loss = F.mse_loss(predicted_q_values, target_q_values)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            q_net.parameters(),
            max_norm=wandb.config.grad_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()

        with torch.no_grad():
            # Evaluate the current agent.
            eval_episode_score, eval_episode_steps = evaluate_episode()

            # Log metrics.
            metrics = {
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "exploration_epsilon": exploration_epsilon,
                "avg_predicted_value": (
                    all_predicted_q_values.amax(dim=1).mean().item()
                    / wandb.config.reward_scale
                ),
                "eval_episode_score": eval_episode_score,
                "eval_episode_steps": eval_episode_steps,
            }
            wandb.log(metrics)

        if iter_num % 500 == 0:
            # Save a checkpoint.
            directory = "checkpoints"
            torch.save(q_net, f"{directory}/q_net-latest.pt")
            shutil.copyfile(
                f"{directory}/q_net-latest.pt",
                f"{directory}/q_net-iter{iter_num:07}.pt",
            )

        # Anneal exploration_epsilon.
        exploration_epsilon = lerp(
            wandb.config.initial_epsilon,
            wandb.config.final_epsilon,
            iter_num / (wandb.config.num_iters - 1),
        )

        # Collect experience.
        generate_experience_step()


@torch.no_grad()
def visualize_agent(reward_scale: float):
    gym = pacbot_rs.PacmanGym(random_start=True)
    gym.reset()

    q_net.eval()

    for step_num in itertools.count(1):
        obs = torch.from_numpy(gym.obs_numpy())
        action_values = q_net(obs.unsqueeze(0)).squeeze(0)
        print(action_values / reward_scale)
        reward, done = gym.step(action_values.argmax())
        print("reward:", reward)

        print()
        print(f"Step {step_num}")
        gym.print_game_state()
        print()

        if done:
            break
        time.sleep(0.2)


try:
    train()
except KeyboardInterrupt:
    pass
reward_scale = wandb.config.reward_scale
wandb.finish()

while True:
    visualize_agent(reward_scale)
    try:
        input("Press enter to view another episode")
    except KeyboardInterrupt:
        print()
        break
