from argparse import ArgumentParser
from collections import deque
import itertools
from pathlib import Path
import random
import shutil
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb
from tqdm import tqdm

from pacbot_rs import AlphaZeroConfig, ExperienceCollector, ExperienceItem, MCTSContext, PacmanGym

import models
from timing import time_block


hyperparam_defaults = {
    "learning_rate": 0.0001,
    "policy_loss_weight": 10.0,
    "num_iters": 10_000,
    "batch_size": 2048,
    "train_instances_per_iter": 2048 * 5,  # Train on this many experience instances per iteration.
    "num_parallel_envs": 128,
    "experience_steps": 500,  # Collect this many steps of experience each iteration (on average).
    "experience_buffer_size": 20_000,
    "tree_size": 100,
    "max_episode_length": 1000,
    "discount_factor": 0.99,
    "evaluate_iters": 1,  # Evaluate every ___ iterations.
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
    group="alphazero",
    config={
        "device": str(device),
        **{name: getattr(args, name) for name in hyperparam_defaults.keys()},
    },
    mode="disabled" if args.eval or args.no_wandb else "online",
)


# Initialize the network.
obs_shape = PacmanGym(random_start=True).obs_numpy().shape
num_actions = 5
model_class = getattr(models, wandb.config.model)
model = model_class(obs_shape, num_actions + 1).to(device)
print(f"model has {sum(p.numel() for p in model.parameters())} parameters")

has_model_updated = False


def model_evaluator(obs: np.ndarray, action_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if has_model_updated:
        # Get value and policy predictions from the model.
        obs = torch.from_numpy(obs).to(device)
        action_mask = torch.from_numpy(action_mask).to(device)

        predictions = model(obs)
        values = predictions[:, -1]
        policy_logits = predictions[:, :-1]
        policy_logits[~action_mask] = -torch.inf

        policy_dist = Categorical(logits=policy_logits)
        policy_probs = policy_dist.probs

        return values.numpy(force=True), policy_probs.numpy(force=True)
    else:
        # The model is freshly-initialized, so just return uniform predictions for speed.
        value = np.zeros(obs.shape[0], dtype=np.float32)
        policy = action_mask / action_mask.sum(axis=-1, keepdims=True, dtype=np.float32)
        return value, policy


@torch.no_grad()
def evaluate_episode(max_steps: int = 1000, greedy: bool = True) -> tuple[int, int]:
    """
    Performs a single evaluation episode.

    Returns (score, total_steps).
    """
    model.eval()

    env = PacmanGym(random_start=True)
    env.reset()
    if greedy:
        for step_num in range(1, max_steps + 1):
            _, policy = model_evaluator(env.obs_numpy()[None], np.array([env.action_mask()]))
            action = policy.squeeze(0).argmax().item()
            _, done = env.step(action)

            if done:
                break

        return env.score(), step_num
    else:
        mc = MCTSContext(env, model_evaluator)
        mc.reset()

        for step_num in range(1, max_steps + 1):
            action = mc.ponder_and_choose(wandb.config.tree_size)
            _, done = mc.take_action(action)

            if done:
                break

        return mc.env.score(), step_num


def train():
    global has_model_updated

    # Initialize the experience collector and buffer.
    exp_collector = ExperienceCollector(
        model_evaluator,
        AlphaZeroConfig(
            tree_size=wandb.config.tree_size,
            max_episode_length=wandb.config.max_episode_length,
            discount_factor=wandb.config.discount_factor,
            num_parallel_envs=wandb.config.num_parallel_envs,
        ),
    )
    exp_buffer = deque[ExperienceItem](maxlen=wandb.config.experience_buffer_size)
    num_exp_steps_needed = exp_buffer.maxlen  # Start by filling the buffer.

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    num_train_instances_needed = 0

    for iter_num in tqdm(range(wandb.config.num_iters), smoothing=0.1):
        with torch.no_grad():
            model.eval()
            num_exp_steps_needed = min(
                num_exp_steps_needed + wandb.config.experience_steps,
                exp_buffer.maxlen,
            )
            exp_progress = tqdm(desc="Collect experience", unit="step", total=num_exp_steps_needed)
            while num_exp_steps_needed > 0:
                # Generate some experience.
                new_exp = exp_collector.generate_experience()
                num_exp_steps_needed -= len(new_exp)

                # Add it to the buffer.
                exp_buffer.extend(new_exp)

                # Update the progress bar.
                exp_progress.update(len(new_exp))
            exp_progress.close()

        value_losses = []
        policy_losses = []
        avg_entropies = []
        losses = []
        grad_norms = []
        num_train_instances_needed += wandb.config.train_instances_per_iter
        while num_train_instances_needed > 0:
            batch = random.sample(exp_buffer, k=wandb.config.batch_size)
            num_train_instances_needed -= wandb.config.batch_size
            with time_block("Collate batch"):
                # Collate the batch.
                with device:
                    obs_batch = torch.stack([torch.from_numpy(item.obs) for item in batch])
                    obs_batch = obs_batch.to(device)
                    action_mask_batch = torch.tensor([item.action_mask for item in batch])
                    value_target_batch = torch.tensor([item.value for item in batch])
                    value_target_batch *= wandb.config.reward_scale
                    policy_target_batch = torch.tensor([item.action_distribution for item in batch])

            with time_block("Compute loss and update parameters"):
                # Compute the model's predictions.
                model.train()
                predictions = model(obs_batch)
                predicted_values = predictions[:, -1]
                policy_logits = torch.where(action_mask_batch, predictions[:, :-1], -torch.inf)

                # Compute the value loss (MSE).
                value_loss = F.mse_loss(predicted_values, value_target_batch)

                # Compute the policy loss (masked cross entropy).
                log_denominator = torch.logsumexp(policy_logits, dim=-1, keepdim=True)
                unmasked_losses = policy_target_batch * (policy_logits - log_denominator)
                policy_loss = -torch.where(action_mask_batch, unmasked_losses, 0.0).sum(-1).mean()

                # Compute the overall loss.
                loss = value_loss + (wandb.config.policy_loss_weight * policy_loss)

                # Compute the gradient and update the parameters.
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=wandb.config.grad_clip_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()

                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                avg_entropies.append(Categorical(logits=policy_logits).entropy().mean().item())
                losses.append(loss.item())
                grad_norms.append(grad_norm.item())
                has_model_updated = True

        with torch.no_grad():
            # Log metrics.
            metrics = {
                "avg_value_loss": np.mean(value_losses),
                "avg_policy_loss": np.mean(policy_losses),
                "avg_policy_entropy": np.mean(avg_entropies),
                "avg_loss": np.mean(losses),
                "max_grad_norm": max(grad_norms),
                "avg_predicted_value": predicted_values.mean() / wandb.config.reward_scale,
                "avg_target_value": value_target_batch.mean() / wandb.config.reward_scale,
            }
            if iter_num % wandb.config.evaluate_iters == 0:
                with time_block("Evaluate the current agent"):
                    # Evaluate the current agent.
                    episode_score, episode_steps = evaluate_episode()
                    metrics.update(
                        eval_episode_score=episode_score,
                        eval_episode_steps=episode_steps,
                    )
            wandb.log(metrics)

        if iter_num % 10 == 0:
            with time_block("Save checkpoint"):
                # Save a checkpoint.
                checkpoint_dir = Path(args.checkpoint_dir)
                checkpoint_dir.mkdir(exist_ok=True)
                torch.save(model, checkpoint_dir / "model-latest.pt")
                shutil.copyfile(
                    checkpoint_dir / "model-latest.pt",
                    checkpoint_dir / f"model-iter{iter_num:07}.pt",
                )


@torch.no_grad()
def visualize_agent():
    model.eval()

    env = PacmanGym(random_start=True)
    env.reset()
    mc = MCTSContext(env, model_evaluator)
    mc.reset()

    print()
    print(f"Step 0")
    mc.env.print_game_state()
    print()

    start = time.perf_counter()
    for step_num in itertools.count(1):
        elapsed = time.perf_counter() - start
        sleep_time = max(0, 0.2 - elapsed)
        # print(sleep_time)
        time.sleep(sleep_time)
        start = time.perf_counter()

        action = mc.ponder_and_choose(args.tree_size)
        with np.printoptions(precision=4, suppress=True):
            print(f"Policy prior:        {np.array(mc.policy_prior())}")
            print(f"Action distribution: {np.array(mc.action_distribution())}  =>  {action}")
        reward, done = mc.take_action(action)
        print("reward:", reward)

        print()
        print(f"Step {step_num}")
        mc.env.print_game_state()
        print()

        if done:
            break


if args.eval:
    model = torch.load(args.eval, map_location=device)
    has_model_updated = True
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
