import random
from typing import Iterator, NamedTuple

import torch
from torch.distributions import Categorical
import numpy as np

import pacbot_rs

from models import NetV2


class StepExperienceItems(NamedTuple):
    observations: torch.Tensor
    actions: torch.IntTensor
    log_old_action_probs: torch.FloatTensor
    rewards: torch.FloatTensor
    dones: torch.BoolTensor


class ExperienceTrainingItem(NamedTuple):
    obs: torch.Tensor
    action: torch.IntTensor  # scalar
    log_old_action_prob: torch.FloatTensor  # scalar
    return_: torch.FloatTensor  # scalar
    advantage: torch.FloatTensor  # scalar


class ExperienceBuffer:
    """
    Handles gathering experience from an environment instance and storing it in an experience buffer.
    """

    def __init__(
        self,
        policy_net: NetV2,
        value_net: NetV2,
        num_parallel_envs: int,
        discount_factor: float,
        device: torch.device = torch.device("cpu"),
        gae_lambda: float = 0.95,
        reward_scale: float = 1.0,
    ) -> None:
        self._buffer = list[StepExperienceItems]()
        self.policy_net = policy_net
        self.value_net = value_net
        self.discount_factor = discount_factor
        self.device = device
        self.gae_lambda = gae_lambda
        self.reward_scale = reward_scale

        # Initialize the environments.
        self._envs = [pacbot_rs.PacmanGym(random_start=True) for _ in range(num_parallel_envs)]
        for env in self._envs:
            env.reset()
        self._last_obs = self._make_current_obs()

    def _make_current_obs(self) -> torch.Tensor:
        obs = [env.obs_numpy() for env in self._envs]
        return torch.from_numpy(np.stack(obs)).to(self.device)

    @property
    def obs_shape(self) -> torch.Size:
        return self._last_obs.shape[1:]

    @property
    def num_actions(self) -> int:
        return 5

    def clear(self) -> None:
        """Removes all experience from the buffer. Does not affect the current environment states."""
        self._buffer.clear()

    @torch.no_grad()
    def generate_experience_step(self) -> None:
        """Generates one step of experience for each parallel env and adds them to the buffer."""

        # Choose an action using the provided policy.
        action_masks = [env.action_mask() for env in self._envs]
        action_masks = torch.from_numpy(np.stack(action_masks)).to(self.device)
        action_logits = self.policy_net(self._last_obs)
        action_logits[~action_masks] = -torch.inf
        action_dist = Categorical(logits=action_logits)
        actions = action_dist.sample()
        log_action_probs = action_dist.log_prob(actions)

        next_obs_stack = []
        rewards = []
        dones = []
        for env, action in zip(self._envs, actions.tolist()):
            # Perform the action and observe the transition.
            reward, done = env.step(action)
            reward *= self.reward_scale
            rewards.append(reward)
            dones.append(done)
            if done:
                next_obs = None
            else:
                next_obs = torch.from_numpy(env.obs_numpy()).to(self.device)

            # Reset the environment if necessary and update last_obs.
            if next_obs is None:
                env.reset()
                next_obs_stack.append(torch.from_numpy(env.obs_numpy()).to(self.device))
            else:
                next_obs_stack.append(next_obs)

        self._buffer.append(
            StepExperienceItems(
                self._last_obs,
                actions,
                log_action_probs,
                torch.tensor(rewards, device=self.device),
                torch.tensor(dones, device=self.device),
            )
        )
        self._last_obs = torch.stack(next_obs_stack)

    @torch.no_grad()
    def compute_training_items(self) -> None:
        self._train_buffer = list[ExperienceTrainingItem]()
        next_values = self.value_net(self._last_obs).squeeze(dim=1)
        returns = next_values
        advantages = 0
        for exp_items in reversed(self._buffer):
            # Set the next value, advantage, and return to zero for `done` transitions.
            next_values = torch.where(exp_items.dones, 0, next_values)
            advantages = torch.where(exp_items.dones, 0, advantages)
            returns = torch.where(exp_items.dones, 0, returns)

            # Compute the advantages.
            values = self.value_net(exp_items.observations).squeeze(dim=1)
            td_errors = (exp_items.rewards + self.discount_factor * next_values) - values
            advantages = td_errors + (self.discount_factor * self.gae_lambda * advantages)

            # Compute the returns.
            returns = exp_items.rewards + self.discount_factor * returns

            # Add complete experience items to the training buffer.
            self._train_buffer.extend(
                ExperienceTrainingItem(*fields)
                for fields in zip(
                    exp_items.observations,
                    exp_items.actions,
                    exp_items.log_old_action_probs,
                    returns,
                    advantages,
                )
            )

    def batches(self, batch_size: int, num_epochs: int) -> Iterator[list[ExperienceTrainingItem]]:
        """
        Returns an iterator over batches of transitions from the buffer.
        The entire buffer will be repeated num_epochs times, shuffling before each epoch.
        """
        for _ in range(num_epochs):
            random.shuffle(self._train_buffer)
            for i in range(0, len(self._train_buffer), batch_size):
                yield self._train_buffer[i : i + batch_size]
