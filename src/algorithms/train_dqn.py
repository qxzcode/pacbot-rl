import copy
import itertools
import shutil
import time
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

import pacbot_rs

from models import QNet
from policies import EpsilonGreedy, MaxQPolicy
from replay_buffer import ReplayBuffer
from timing import time_block
from utils import lerp


wandb.init(
    project="pacbot-dqn",
    config={
        "learning_rate": 0.001,
        "batch_size": 512,
        "num_iters": 150_000,
        "replay_buffer_size": 10_000,
        "target_network_update_steps": 500,  # Update the target network every ___ steps.
        "evaluate_steps": 1,  # Evaluate every ___ steps.
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "discount_factor": 0.99,
        "reward_scale": 1 / 50,
        "grad_clip_norm": 0.1,
    },
    # mode="disabled",
)


# Initialize the Q network.
obs_shape = pacbot_rs.PacmanGym(random_start=True).obs_numpy().shape
num_actions = 5
q_net = QNet(obs_shape, num_actions)
print(f"q_net has {sum(p.numel() for p in q_net.parameters())} parameters")


# Initialize the replay buffer.
replay_buffer = ReplayBuffer(
    maxlen=wandb.config.replay_buffer_size,
    policy=EpsilonGreedy(MaxQPolicy(q_net), num_actions, wandb.config.initial_epsilon),
)
replay_buffer.fill()


@torch.no_grad()
def evaluate_episode(max_steps: int = 1000) -> tuple[int, int]:
    """
    Performs a single evaluation episode.

    Returns (score, total_steps).
    """
    gym = pacbot_rs.PacmanGym(random_start=True)
    gym.reset()

    q_net.eval()
    policy = MaxQPolicy(q_net)

    for step_num in range(1, max_steps + 1):
        obs = torch.from_numpy(gym.obs_numpy())
        _, done = gym.step(policy(obs))

        if done:
            break

    return (gym.score(), step_num)


def train():
    optimizer = torch.optim.Adam(q_net.parameters(), lr=wandb.config.learning_rate)

    for iter_num in tqdm(range(wandb.config.num_iters), smoothing=0.01):
        if iter_num % wandb.config.target_network_update_steps == 0:
            with time_block("Update target network"):
                # Update the target network.
                target_q_net = copy.deepcopy(q_net)
                target_q_net.eval()

        with time_block("Collate batch"):
            # Sample and collate a batch.
            batch = replay_buffer.sample_batch(wandb.config.batch_size)
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

        with time_block("Compute target Q values"):
            # Get the target Q values.
            with torch.no_grad():
                returns = target_q_net(next_obs_batch).amax(dim=1)
                discounted_returns = wandb.config.discount_factor * returns
                discounted_returns[done_mask] = 0.0
                target_q_values = reward_batch + discounted_returns

        with time_block("Compute loss and update parameters"):
            # Compute the loss and update the parameters.
            with time_block("optimizer.zero_grad()"):
                q_net.train()
                optimizer.zero_grad()

            with time_block("Forward pass"):
                all_predicted_q_values = q_net(obs_batch)
                predicted_q_values = all_predicted_q_values[
                    range(len(batch)), action_batch
                ]
                loss = F.mse_loss(predicted_q_values, target_q_values)

            with time_block("Backward pass"):
                loss.backward()
            with time_block("Clip grad norm"):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    q_net.parameters(),
                    max_norm=wandb.config.grad_clip_norm,
                    error_if_nonfinite=True,
                )
            with time_block("optimizer.step()"):
                optimizer.step()

        with torch.no_grad():
            # Log metrics.
            metrics = {
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "exploration_epsilon": replay_buffer.policy.epsilon,
                "avg_predicted_value": (
                    all_predicted_q_values.amax(dim=1).mean().item()
                    / wandb.config.reward_scale
                ),
            }
            if iter_num % wandb.config.evaluate_steps == 0:
                with time_block("Evaluate the current agent"):
                    # Evaluate the current agent.
                    eval_episode_score, eval_episode_steps = evaluate_episode()
                    metrics.update(
                        eval_episode_score=eval_episode_score,
                        eval_episode_steps=eval_episode_steps,
                    )
            wandb.log(metrics)

        if iter_num % 500 == 0:
            with time_block("Save checkpoint"):
                # Save a checkpoint.
                directory = "checkpoints2"
                torch.save(q_net, f"{directory}/q_net-latest.pt")
                shutil.copyfile(
                    f"{directory}/q_net-latest.pt",
                    f"{directory}/q_net-iter{iter_num:07}.pt",
                )

        # Anneal the exploration policy's epsilon.
        replay_buffer.policy.epsilon = lerp(
            wandb.config.initial_epsilon,
            wandb.config.final_epsilon,
            iter_num / (wandb.config.num_iters - 1),
        )

        # Collect experience.
        with time_block("Collect experience"):
            replay_buffer.generate_experience_step()


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


do_training = True
if do_training:
    try:
        train()
    except KeyboardInterrupt:
        pass
    reward_scale = wandb.config.reward_scale
    wandb.finish()
else:
    reward_scale = 1 / 50
    q_net = torch.load("checkpoints/q_net-latest.pt")

while True:
    visualize_agent(reward_scale)
    try:
        input("Press enter to view another episode")
    except KeyboardInterrupt:
        print()
        break
