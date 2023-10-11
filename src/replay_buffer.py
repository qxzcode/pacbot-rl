from collections import deque
import random
from typing import Callable, Generic, NamedTuple, Optional, TypeVar

import torch

import pacbot_rs


class ReplayItem(NamedTuple):
    obs: torch.Tensor
    action: int
    reward: int
    next_obs: Optional[torch.Tensor]


P = TypeVar("P", bound=Callable[[torch.Tensor], int])


class ReplayBuffer(Generic[P]):
    """
    Handles gathering experience from an environment instance and storing it in a replay buffer.
    """

    def __init__(
        self,
        maxlen: int,
        policy: P,
    ) -> None:
        self._buffer = deque[ReplayItem](maxlen=maxlen)
        self.policy = policy

        # Initialize the environment.
        self._gym = pacbot_rs.PacmanGym(random_start=True)
        self._gym.reset()
        self._last_obs = torch.from_numpy(self._gym.obs_numpy())

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

        # Choose an action (using q_net and epsilon-greedy for exploration).
        # TODO: invalid action masking?
        action = self.policy(self._last_obs)

        # Perform the action and observe the transition.
        reward, done = self._gym.step(action)
        next_obs = None if done else torch.from_numpy(self._gym.obs_numpy())

        # Add the transition to the replay buffer.
        self._buffer.append(ReplayItem(self._last_obs, action, reward, next_obs))

        # Reset the environment if necessary and update last_obs.
        if next_obs is None:
            self._gym.reset()
            self._last_obs = torch.from_numpy(self._gym.obs_numpy())
        else:
            self._last_obs = next_obs

    def sample_batch(self, batch_size: int) -> list[ReplayItem]:
        """Samples a batch of transitions from the buffer."""
        return random.sample(self._buffer, k=batch_size)
