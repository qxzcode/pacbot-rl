import random
from typing import Callable, Generic, TypeVar

import torch

from models import QNet


P = TypeVar("P", bound=Callable[[torch.Tensor], int])


class EpsilonGreedy(Generic[P]):
    """ϵ-greedy policy that wraps another policy."""

    def __init__(self, original_policy: P, num_actions: int, epsilon: float) -> None:
        self.original_policy = original_policy
        self.num_actions = num_actions
        self.epsilon = epsilon

    def __call__(self, obs: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            return self.original_policy(obs)


class MaxQPolicy:
    """
    A policy that selects the action with the highest Q(s, a) value, as predicted by a Q network.

    The Q network is expected to take a (batched) observation tensor and return a (batched) vector
    of Q values, with shape (1, num_actions).
    """

    def __init__(self, q_net: QNet) -> None:
        self.q_net = q_net

    def __call__(self, obs: torch.Tensor) -> int:
        action_values = self.q_net(obs.unsqueeze(0)).squeeze(0)
        return action_values.argmax()
