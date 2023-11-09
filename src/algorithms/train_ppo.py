from argparse import ArgumentParser
import itertools
from pathlib import Path
import shutil
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb
from tqdm import tqdm

from pacbot_rs import PacmanGym

import models
from policies import PNetPolicy
from ppo_experience_buffer import ExperienceBuffer
from timing import time_block


hyperparam_defaults = {
    "value_learning_rate": 0.001,
    "policy_learning_rate": 0.0001,
    "batch_size": 2048,
    "num_iters": 10_000,
    "num_train_iters": 4,
    "num_parallel_envs": 128,
    "experience_steps": 200,  # Collect this many steps of experience (per parallel env) each iteration.
    "evaluate_iters": 1,  # Evaluate every ___ iterations.
    "discount_factor": 0.99,
    "gae_lambda": 0.95,
    "ppo_epsilon": 0.2,
    "reward_scale": 1 / 100,
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

    Returns (score, total_steps, avg_action_entropy).
    """
    gym = PacmanGym(random_start=True)
    gym.reset()

    policy_net.eval()
    policy = PNetPolicy(policy_net)

    entropies = []
    for step_num in range(1, max_steps + 1):
        obs = torch.from_numpy(gym.obs_numpy()).to(device).unsqueeze(0)
        action_mask = torch.tensor(gym.action_mask(), device=device).unsqueeze(0)
        action, entropy = policy.action_and_entropy(obs, action_mask)
        entropies.append(entropy.item())
        _, done = gym.step(action.item())

        if done:
            break

    return gym.score(), step_num, np.mean(entropies)


def train():
    # Initialize the replay buffer.
    exp_buffer = ExperienceBuffer(
        policy_net=policy_net,
        value_net=value_net,
        num_parallel_envs=wandb.config.num_parallel_envs,
        discount_factor=wandb.config.discount_factor,
        device=device,
        gae_lambda=wandb.config.gae_lambda,
        reward_scale=wandb.config.reward_scale,
    )

    # Initialize the optimizers.
    value_lr = wandb.config.value_learning_rate
    policy_lr = wandb.config.policy_learning_rate
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_lr)
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=policy_lr)

    for iter_num in tqdm(range(wandb.config.num_iters), smoothing=0.01):
        with time_block("Collect experience"):
            policy_net.eval()
            value_net.eval()
            exp_buffer.clear()
            for _ in range(wandb.config.experience_steps):
                exp_buffer.generate_experience_step()
            exp_buffer.compute_training_items()

        total_value_loss = 0
        total_policy_loss = 0
        value_grad_norms = []
        policy_grad_norms = []
        for batch in exp_buffer.batches(wandb.config.batch_size, wandb.config.num_train_iters):
            with time_block("Collate batch"):
                # Sample and collate a batch.
                with device:
                    obs_batch = torch.stack([item.obs for item in batch])
                    action_batch = torch.stack([item.action for item in batch])
                    log_old_action_prob_batch = torch.stack(
                        [item.log_old_action_prob for item in batch]
                    )
                    return_batch = torch.stack([item.return_ for item in batch])
                    advantage_batch = torch.stack([item.advantage for item in batch])

            with time_block("Compute loss and update parameters (value net)"):
                value_net.train()

                # Compute the loss.
                predicted_returns = value_net(obs_batch).squeeze(dim=1)
                value_loss = F.mse_loss(predicted_returns, return_batch)
                total_value_loss += value_loss.item()

                # Compute the gradient and update the parameters.
                value_optimizer.zero_grad()
                value_loss.backward()
                value_grad_norm = torch.nn.utils.clip_grad_norm_(
                    value_net.parameters(),
                    max_norm=wandb.config.grad_clip_norm,
                    error_if_nonfinite=True,
                )
                value_grad_norms.append(value_grad_norm.item())
                value_optimizer.step()

            with time_block("Compute loss and update parameters (policy net)"):
                policy_net.train()

                # Compute the loss.
                action_logits = policy_net(obs_batch)
                log_new_action_probs = Categorical(logits=action_logits).log_prob(action_batch)
                ratios = (log_new_action_probs - log_old_action_prob_batch).exp()
                unclipped_loss = ratios * advantage_batch
                clipped_loss = (
                    1.0 + wandb.config.ppo_epsilon * advantage_batch.sign()
                ) * advantage_batch
                policy_loss = -torch.min(unclipped_loss, clipped_loss).mean()
                total_policy_loss += policy_loss.item()

                # Compute the gradient and update the parameters.
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy_net.parameters(),
                    max_norm=wandb.config.grad_clip_norm,
                    error_if_nonfinite=True,
                )
                policy_grad_norms.append(policy_grad_norm.item())
                policy_optimizer.step()

        with torch.no_grad():
            # Log metrics.
            metrics = {
                "avg_value_loss": total_value_loss / wandb.config.num_train_iters,
                "max_value_grad_norm": max(value_grad_norms),
                "avg_policy_loss": total_policy_loss / wandb.config.num_train_iters,
                "max_policy_grad_norm": max(policy_grad_norms),
                "avg_predicted_value": predicted_returns.mean() / wandb.config.reward_scale,
                "avg_target_value": return_batch.mean() / wandb.config.reward_scale,
            }
            if iter_num % wandb.config.evaluate_iters == 0:
                with time_block("Evaluate the current agent"):
                    # Evaluate the current agent.
                    episode_score, episode_steps, avg_policy_entropy = evaluate_episode()
                    metrics.update(
                        eval_episode_score=episode_score,
                        eval_episode_steps=episode_steps,
                        eval_avg_policy_entropy=avg_policy_entropy,
                    )
            wandb.log(metrics)

        if iter_num % 30 == 0:
            with time_block("Save checkpoint"):
                # Save a checkpoint.
                checkpoint_dir = Path(args.checkpoint_dir)
                checkpoint_dir.mkdir(exist_ok=True)
                torch.save(policy_net, checkpoint_dir / "policy_net-latest.pt")
                shutil.copyfile(
                    checkpoint_dir / "policy_net-latest.pt",
                    checkpoint_dir / f"policy_net-iter{iter_num:07}.pt",
                )
                torch.save(value_net, checkpoint_dir / "value_net-latest.pt")
                shutil.copyfile(
                    checkpoint_dir / "value_net-latest.pt",
                    checkpoint_dir / f"value_net-iter{iter_num:07}.pt",
                )


@torch.no_grad()
def visualize_agent():
    gym = PacmanGym(random_start=True)
    gym.reset()

    policy_net.eval()

    print()
    print(f"Step 0")
    gym.print_game_state()
    print()

    for step_num in itertools.count(1):
        time.sleep(0.2)

        obs = torch.from_numpy(gym.obs_numpy()).to(device)
        action_logits = policy_net(obs.unsqueeze(0)).squeeze(0)
        action_logits[~torch.tensor(gym.action_mask())] = -torch.inf
        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample().item()
        with np.printoptions(precision=4, suppress=True):
            print(f"Action probabilities: {action_dist.probs.numpy(force=True)}  =>  {action}")
        reward, done = gym.step(action)
        print("reward:", reward)

        print()
        print(f"Step {step_num}")
        gym.print_game_state()
        print()

        if done:
            break


if args.eval:
    policy_net = torch.load(args.eval, map_location=device)
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
