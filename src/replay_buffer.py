from collections import deque
import random
from typing import Generic, NamedTuple, Optional, TypeVar

import torch

import pacbot_rs
from debug_probe_envs import *

from policies import Policy


class ReplayItem(NamedTuple):
    obs: torch.Tensor
    action: int
    reward: int
    next_obs: Optional[torch.Tensor]
    next_action_mask: list[bool]


P = TypeVar("P", bound=Policy)


# DebugProbeGym = lambda: ConstantRewardSequenceProbeGym([0] * 100 + [50])
# DebugProbeGym = lambda: PredictDelayedRewardProbeGym(
#     1, keep_giving_answer=False, tell_if_incorrect=False
# )
DebugProbeGym = CartPoleGym


class ReplayBuffer(Generic[P]):
    """
    Handles gathering experience from an environment instance and storing it in a replay buffer.
    """

    def __init__(
        self,
        maxlen: int,
        policy: P,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._buffer = deque[ReplayItem](maxlen=maxlen)
        self.policy = policy
        self.device = device

        # Initialize the environment.
        self._gym = pacbot_rs.PacmanGym(random_start=True)
        # self._gym = DebugProbeGym()
        self._gym.reset()
        self._last_obs = torch.from_numpy(self._gym.obs_numpy()).to(self.device)

    @property
    def obs_shape(self) -> torch.Size:
        return self.last_obs.shape

    @property
    def num_actions(self) -> int:
        return 5

    def fill(self) -> None:
        """Generates experience until the buffer is filled to capacity."""
        while len(self._buffer) < self._buffer.maxlen:
            self.generate_experience_step()

    @torch.no_grad()
    def generate_experience_step(self) -> None:
        """Generates one step of experience and adds it to the buffer."""

        # Choose an action using the provided policy.
        action = self.policy(self._last_obs, self._gym.action_mask())

        # Perform the action and observe the transition.
        reward, done = self._gym.step(action)
        if done:
            next_obs = None
            next_action_mask = [False] * self.num_actions
        else:
            next_obs = torch.from_numpy(self._gym.obs_numpy()).to(self.device)
            next_action_mask = self._gym.action_mask()

        # Add the transition to the replay buffer.
        self._buffer.append(ReplayItem(self._last_obs, action, reward, next_obs, next_action_mask))

        # Reset the environment if necessary and update last_obs.
        if next_obs is None:
            self._gym.reset()
            self._last_obs = torch.from_numpy(self._gym.obs_numpy()).to(self.device)
        else:
            self._last_obs = next_obs

    def sample_batch(self, batch_size: int) -> list[ReplayItem]:
        """Samples a batch of transitions from the buffer."""
        return random.sample(self._buffer, k=batch_size)
