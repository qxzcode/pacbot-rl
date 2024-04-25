from argparse import ArgumentParser
from contextlib import nullcontext
import copy
import itertools
from pathlib import Path
import shutil
import time
import numpy as np

import safetensors.torch
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm


from pacbot_rs import PacmanGym

import models
from policies import EpsilonGreedy, MaxQPolicy
from replay_buffer import ReplayBuffer, reset_env
from timing import time_block
from utils import lerp


hyperparam_defaults = {
    "learning_rate": 0.0001,
    "batch_size": 512,
    "num_iters": 2_000_000,
    "replay_buffer_size": 10_000,
    "num_parallel_envs": 32,
    "random_start_proportion": 0.5,
    "experience_steps": 4,
    "target_network_update_steps": 500,  # Update the target network every ___ steps.
    "evaluate_steps": 10,  # Evaluate every ___ steps.
    "initial_epsilon": 0.1,
    "final_epsilon": 0.1 * 0.05,
    "discount_factor": 0.99,
    "reward_scale": 1 / 50,
    "grad_clip_norm": 10_000,
    "model": "QNetV2",
}

parser = ArgumentParser()
parser.add_argument("--eval", metavar="CHECKPOINT", default=None)
parser.add_argument("--finetune", metavar="CHECKPOINT", default=None)
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
    tags=["DQN"] + (["finetuning"] if args.finetune else []),
    config={
        "device": str(device),
        **{name: getattr(args, name) for name in hyperparam_defaults.keys()},
    },
    mode="disabled" if args.eval or args.no_wandb else "online",
)


# Initialize the Q network.
obs_shape = PacmanGym(random_start=True, random_ticks=True).obs_numpy().shape
num_actions = 5
model_class = getattr(models, wandb.config.model)
q_net = model_class(obs_shape, num_actions).to(device)
print(f"q_net has {sum(p.numel() for p in q_net.parameters())} parameters")
if args.finetune:
    q_net = torch.load(args.finetune, map_location=device)
    print(f"Finetuning from parameters from {args.finetune}")


@torch.no_grad()
def evaluate_episode(max_steps: int = 1000) -> tuple[int, int, bool]:
    """
    Performs a single evaluation episode.

    Returns (score, total_steps, is_board_cleared).
    """
    gym = PacmanGym(random_start=False, random_ticks=False)
    reset_env(gym)

    q_net.eval()
    policy = MaxQPolicy(q_net)

    for step_num in range(1, max_steps + 1):
        obs = torch.from_numpy(gym.obs_numpy()).to(device).unsqueeze(0)
        action_mask = torch.tensor(gym.action_mask(), device=device).unsqueeze(0)
        _, done = gym.step(policy(obs, action_mask).item())

        if done:
            break

    return (gym.score(), step_num, done and gym.lives() == 3)


def train():
    # Initialize the replay buffer.
    replay_buffer = ReplayBuffer(
        maxlen=wandb.config.replay_buffer_size,
        policy=EpsilonGreedy(
            MaxQPolicy(q_net),
            num_actions,
            wandb.config.initial_epsilon if args.finetune else 1.0,
        ),
        num_parallel_envs=wandb.config.num_parallel_envs,
        random_start_proportion=wandb.config.random_start_proportion,
        device=device,
    )
    replay_buffer.fill()
    replay_buffer.policy.epsilon = wandb.config.initial_epsilon

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(q_net.parameters(), lr=wandb.config.learning_rate)

    # Automatic Mixed Precision stuff.
    use_amp = False  # device.type == "cuda"
    grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.float16) if use_amp else nullcontext()
    )

    for iter_num in tqdm(range(wandb.config.num_iters), smoothing=0.01):
        if iter_num % wandb.config.target_network_update_steps == 0:
            with time_block("Update target network"):
                # Update the target network.
                target_q_net = copy.deepcopy(q_net)
                target_q_net.eval()

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
                with autocast:
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

        with time_block("Compute loss and update parameters"):
            # Compute the loss and update the parameters.
            with time_block("optimizer.zero_grad()"):
                q_net.train()
                optimizer.zero_grad()

            with time_block("Forward pass"):
                with autocast:
                    all_predicted_q_values = q_net(obs_batch)
                    predicted_q_values = all_predicted_q_values[range(len(batch)), action_batch]
                    loss = F.mse_loss(predicted_q_values, target_q_values)

            with time_block("Backward pass"):
                grad_scaler.scale(loss).backward()
            with time_block("Clip grad norm"):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    q_net.parameters(),
                    max_norm=wandb.config.grad_clip_norm,
                    error_if_nonfinite=True,
                )
            with time_block("Step optimizer"):
                grad_scaler.step(optimizer)
                grad_scaler.update()

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
            if iter_num % wandb.config.evaluate_steps == 0:
                with time_block("Evaluate the current agent"):
                    # Evaluate the current agent.
                    eval_episode_score, eval_episode_steps, cleared_board = evaluate_episode()
                    metrics.update(
                        eval_episode_score=eval_episode_score,
                        eval_episode_steps=eval_episode_steps,
                        cleared_board=int(cleared_board),
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

                if iter_num % 10_000 == 0:
                    # Log a .safetensors checkpoint to WandB.
                    safetensors_path = checkpoint_dir / "q_net-latest.safetensors"
                    safetensors.torch.save_file(q_net.state_dict(), safetensors_path)
                    wandb.log_artifact(safetensors_path, type="model")

        # Anneal the exploration policy's epsilon.
        replay_buffer.policy.epsilon = lerp(
            wandb.config.initial_epsilon,
            wandb.config.final_epsilon,
            iter_num / (wandb.config.num_iters - 1),
        )

        # Collect experience.
        with time_block("Collect experience"):
            for _ in range(wandb.config.experience_steps):
                replay_buffer.generate_experience_step()


@torch.no_grad()
def visualize_agent():
    gym = PacmanGym(random_start=False, random_ticks=False)
    reset_env(gym)

    q_net.eval()

    print()
    print(f"Step 0")
    gym.print_game_state()
    print()

    for step_num in itertools.count(1):
        time.sleep(0.1)

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
