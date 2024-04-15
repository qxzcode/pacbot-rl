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
        num_parallel_envs: int,
        random_start_proportion: float,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._buffer = deque[ReplayItem](maxlen=maxlen)
        self.policy = policy
        self.device = device

        # Initialize the environments.
        self._envs = [
            pacbot_rs.PacmanGym(random_start=i < num_parallel_envs * random_start_proportion)
            for i in range(num_parallel_envs)
        ]
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

    def fill(self) -> None:
        """Generates experience until the buffer is filled to capacity."""
        while len(self._buffer) < self._buffer.maxlen:
            self.generate_experience_step()

    @torch.no_grad()
    def generate_experience_step(self) -> None:
        """Generates one step of experience for each parallel env and adds them to the buffer."""

        # Choose an action using the provided policy.
        action_masks = [env.action_mask() for env in self._envs]
        action_masks = torch.from_numpy(np.stack(action_masks)).to(self.device)
        actions = self.policy(self._last_obs, action_masks)

        next_obs_stack = []
        for env, last_obs, action in zip(self._envs, self._last_obs, actions.tolist()):
            # Perform the action and observe the transition.
            reward, done = env.step(action)
            if done:
                next_obs = None
                next_action_mask = [False] * self.num_actions
            else:
                next_obs = torch.from_numpy(env.obs_numpy()).to(self.device)
                next_action_mask = env.action_mask()

            # # Subsample to focus training on end-game states.
            # keep_prob = 1.0 if env.remaining_pellets() < 140 else 0.1
            # if random.random() < keep_prob:
            #     print(f"{env.remaining_pellets()=}")
            #     # Add the transition to the replay buffer.
            #     item = ReplayItem(last_obs, action, reward, next_obs, next_action_mask)
            #     self._buffer.append(item)
            # Add the transition to the replay buffer.
            self._buffer.append(ReplayItem(last_obs, action, reward, next_obs, next_action_mask))

            # Reset the environment if necessary and update last_obs.
            if next_obs is None:
                env.reset()
                next_obs_stack.append(torch.from_numpy(env.obs_numpy()).to(self.device))
            else:
                next_obs_stack.append(next_obs)

        self._last_obs = torch.stack(next_obs_stack)

    def sample_batch(self, batch_size: int) -> list[ReplayItem]:
        """Samples a batch of transitions from the buffer."""
        return random.sample(self._buffer, k=batch_size)
