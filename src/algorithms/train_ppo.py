from argparse import ArgumentParser
from contextlib import nullcontext
import copy
import itertools
from pathlib import Path
import shutil
import time
import numpy as np

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from pacbot_rs import PacmanGym

import models
from policies import PNetPolicy
from replay_buffer import ReplayBuffer
from timing import time_block
from utils import lerp


hyperparam_defaults = {
    "learning_rate": 0.0001,
    "batch_size": 512,
    "num_iters": 150_000,
    "num_parallel_envs": 64,
    "experience_steps": 100,  # Collect this many steps of experience (per parallel env) each iteration.
    "evaluate_iters": 10,  # Evaluate every ___ iterations.
    "discount_factor": 0.99,
    "reward_scale": 1 / 50,
    "grad_clip_norm": 0.1,
    "model": "NetV2",
}

parser = ArgumentParser()
parser.add_argument("--eval", metavar="CHECKPOINT", default=None)
parser.add_argument("--no-wandb", action="store_true")
parser.add_argument("--checkpoint-dir", default="checkpoints")
parser.add_argument("--device", default=None)
for name, default_value in hyperparam_defaults.items():
    parser.add_argument(
        f"--{name}",
        type=type(default_value),
        default=default_value,
        help="Default: %(default)s",
    )
args = parser.parse_args()
if not hasattr(models, args.model):
    parser.error(f"Invalid --model: {args.model!r} (must be a class in models.py)")

device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")


reward_scale: float = args.reward_scale
wandb.init(
    project="pacbot-ind-study",
    group="ppo",
    config={
        "device": str(device),
        **{name: getattr(args, name) for name in hyperparam_defaults.keys()},
    },
    mode="disabled" if args.eval or args.no_wandb else "online",
)


# Initialize the Q network.
obs_shape = PacmanGym(random_start=True).obs_numpy().shape
num_actions = 5
model_class = getattr(models, wandb.config.model)
policy_net = model_class(obs_shape, num_actions).to(device)
print(f"policy_net has {sum(p.numel() for p in policy_net.parameters())} parameters")
value_net = model_class(obs_shape, 1).to(device)
print(f"value_net has {sum(p.numel() for p in value_net.parameters())} parameters")


@torch.no_grad()
def evaluate_episode(max_steps: int = 1000) -> tuple[int, int]:
    """
    Performs a single evaluation episode.

    Returns (score, total_steps).
    """
    gym = PacmanGym(random_start=True)
    gym.reset()

    policy_net.eval()
    policy = PNetPolicy(policy_net)

    for step_num in range(1, max_steps + 1):
        obs = torch.from_numpy(gym.obs_numpy()).to(device).unsqueeze(0)
        action_mask = torch.tensor(gym.action_mask(), device=device).unsqueeze(0)
        _, done = gym.step(policy(obs, action_mask).item())

        if done:
            break

    return (gym.score(), step_num)


def train():
    # Initialize the replay buffer.
    replay_buffer = ReplayBuffer(
        maxlen=None,
        policy=PNetPolicy(policy_net),
        num_parallel_envs=wandb.config.num_parallel_envs,
        device=device,
    )

    # Initialize the optimizers.
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=wandb.config.learning_rate)
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=wandb.config.learning_rate)

    for iter_num in tqdm(range(wandb.config.num_iters), smoothing=0.01):
        with time_block("Collect experience"):
            policy_net.eval()
            replay_buffer.clear()
            for _ in range(wandb.config.experience_steps):
                replay_buffer.generate_experience_step()

        with time_block("Collate batch"):
            # Sample and collate a batch.
            with device:
                batch = replay_buffer.sample_batch(wandb.config.batch_size)
                obs_batch = torch.stack([item.obs for item in batch])
                next_obs_batch = torch.stack(
                    [
                        torch.zeros(obs_shape) if item.next_obs is None else item.next_obs
                        for item in batch
                    ]
                )
                done_mask = torch.tensor([item.next_obs is None for item in batch])
                next_action_masks = torch.tensor([item.next_action_mask for item in batch])
                action_batch = torch.tensor([item.action for item in batch])
                reward_batch = torch.tensor(
                    [item.reward * wandb.config.reward_scale for item in batch]
                )

        with time_block("Compute target Q values"):
            # Get the target Q values.
            double_dqn = True
            with torch.no_grad():
                next_q_values = target_q_net(next_obs_batch)
                next_q_values[~next_action_masks] = -torch.inf
                if double_dqn:
                    online_next_q_values = q_net(next_obs_batch)
                    online_next_q_values[~next_action_masks] = -torch.inf
                    next_actions = online_next_q_values.argmax(dim=1)
                else:
                    next_actions = next_q_values.argmax(dim=1)
                returns = next_q_values[range(len(batch)), next_actions]
                discounted_returns = wandb.config.discount_factor * returns
                discounted_returns[done_mask] = 0.0
                target_q_values = reward_batch + discounted_returns

        with time_block("Compute loss and update parameters (value net)"):
            value_net.train()

            # Compute the loss.
            all_predicted_q_values = value_net(obs_batch)
            predicted_q_values = all_predicted_q_values[range(len(batch)), action_batch]
            loss = F.mse_loss(predicted_q_values, target_q_values)

            # Compute the gradient and update the parameters.
            value_optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                value_net.parameters(),
                max_norm=wandb.config.grad_clip_norm,
                error_if_nonfinite=True,
            )
            value_optimizer.step()

        # TODO: policy loss + step

        with torch.no_grad():
            # Log metrics.
            metrics = {
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "exploration_epsilon": replay_buffer.policy.epsilon,
                "avg_predicted_value": (
                    all_predicted_q_values.amax(dim=1).mean().item() / wandb.config.reward_scale
                ),
                "avg_target_q_value": target_q_values.mean() / wandb.config.reward_scale,
            }
            if iter_num % wandb.config.evaluate_iters == 0:
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
                checkpoint_dir = Path(args.checkpoint_dir)
                checkpoint_dir.mkdir(exist_ok=True)
                torch.save(q_net, checkpoint_dir / "q_net-latest.pt")
                shutil.copyfile(
                    checkpoint_dir / "q_net-latest.pt",
                    checkpoint_dir / f"q_net-iter{iter_num:07}.pt",
                )


@torch.no_grad()
def visualize_agent():
    gym = PacmanGym(random_start=True)
    gym.reset()

    q_net.eval()

    print()
    print(f"Step 0")
    gym.print_game_state()
    print()

    for step_num in itertools.count(1):
        time.sleep(0.2)

        obs = torch.from_numpy(gym.obs_numpy()).to(device)
        action_values = q_net(obs.unsqueeze(0)).squeeze(0)
        action_values[~torch.tensor(gym.action_mask())] = -torch.inf
        action = action_values.argmax().item()
        with np.printoptions(precision=4, suppress=True):
            print(f"Q values: {(action_values / reward_scale).numpy(force=True)}  =>  {action}")
        reward, done = gym.step(action)
        print("reward:", reward)

        print()
        print(f"Step {step_num}")
        gym.print_game_state()
        print()

        if done:
            break


if args.eval:
    q_net = torch.load(args.eval, map_location=device)
else:
    try:
        train()
    except KeyboardInterrupt:
        pass
    wandb.finish()

while True:
    visualize_agent()
    try:
        input("Press enter to view another episode")
    except KeyboardInterrupt:
        print()
        break
