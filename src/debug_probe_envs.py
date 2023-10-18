import random
from typing import Callable, Optional

import gymnasium as gym
import numpy as np


class FSMGym:
    """An environment defined by an explicit finite state machine."""

    State = tuple[
        str,
        Callable[[], np.ndarray],
        Callable[[int], tuple[int, Optional["State"]]],
    ]

    def __init__(self, start_state_func: Callable[[], "FSMGym.State"]) -> None:
        self._start_state_func = start_state_func
        self.reset()

    def reset(self) -> None:
        self._state: Optional[FSMGym.State] = self._start_state_func()
        assert self._state is not None
        self._score = 0
        self._done = False

    def step(self, action: int) -> tuple[int, bool]:
        reward, self._state = self._state[2](action)
        self._score += reward
        self._done = self._state is None
        return reward, self._done

    def score(self) -> int:
        return self._score

    def is_done(self) -> bool:
        return self._done

    def action_mask(self) -> list[bool]:
        return [True] * 5

    def obs_numpy(self) -> np.ndarray:
        return self._state[1]().astype(np.float32)

    def print_game_state(self) -> None:
        print(f">> FSM environment: {self._state[0] if self._state else '(DONE)'}")
        if self._state:
            print(f">>   observation: {self.obs_numpy()}")


class ConstantRewardProbeGym(FSMGym):
    def __init__(self) -> None:
        super().__init__(
            lambda: (
                "start state",
                lambda: np.zeros(1),
                lambda action: (50, None),
            )
        )


class PredictRewardProbeGym(FSMGym):
    def __init__(self) -> None:
        def make_start_state():
            answer = random.randrange(5)
            obs = np.zeros(5)
            obs[answer] = 1.0
            return (
                f"{answer=}",
                lambda: obs,
                lambda action: (50 if action == answer else 0, None),
            )

        super().__init__(make_start_state)


class ConstantRewardSequenceProbeGym(FSMGym):
    def __init__(self, reward_sequence: list[int]) -> None:
        def make_state(i, reward, next_state):
            obs = np.zeros(len(reward_sequence))
            obs[i] = 1.0
            return (
                f"state{i}",
                lambda: obs,
                lambda action: (reward, next_state),
            )

        next_state = None
        for i, reward in reversed(list(enumerate(reward_sequence))):
            next_state = make_state(i, reward, next_state)

        super().__init__(lambda: next_state)


def _delay_state(
    delay: int, final_state: FSMGym.State, make_obs: Callable[[int], np.ndarray]
) -> FSMGym.State:
    """Returns a state that delays the transition to final_state by the given number of steps."""
    if delay == 0:
        return final_state
    else:
        return (
            f"delay({delay}, {final_state[0]})",
            lambda: make_obs(delay),
            lambda action: (0, _delay_state(delay - 1, final_state, make_obs)),
        )


class PredictDelayedRewardProbeGym(FSMGym):
    def __init__(self, delay: int, keep_giving_answer: bool, tell_if_incorrect: bool) -> None:
        assert delay > 0

        def make_start_state():
            answer = random.randrange(5)
            obs = np.zeros(5 + (delay + 1))
            obs[answer] = 1.0
            obs[5] = 1.0

            def make_obs(steps_left: int, incorrect: bool = False) -> np.ndarray:
                obs = np.zeros(5 + (delay + 1))
                obs[(5 + delay) - steps_left] = 1.0
                if incorrect:
                    obs[0:5] = 1.0
                elif keep_giving_answer:
                    obs[answer] = 1.0
                return obs

            reward_state = (
                "reward state",
                lambda: make_obs(0),
                lambda action: (50, None),
            )
            incorrect_state = (
                "incorrect state",
                lambda: make_obs(0, incorrect=tell_if_incorrect),
                lambda action: (0, None),
            )
            return (
                f"{answer=}",
                lambda: obs,
                lambda action: (
                    0,
                    _delay_state(
                        delay - 1,
                        reward_state if action == answer else incorrect_state,
                        make_obs,
                    ),
                ),
            )

        super().__init__(make_start_state)


class CartPoleGym:
    """A wrapper around CartPole-v1."""

    def __init__(self) -> None:
        self._env = gym.make("CartPole-v1")  # , render_mode="human")
        self.reset()

    def reset(self) -> None:
        self._obs, info = self._env.reset()
        self._score = 0
        self._done = False

    def step(self, action: int) -> tuple[int, bool]:
        if type(action) is not int:
            action = int(action)

        observation, reward, terminated, truncated, info = self._env.step(action)

        self._obs = observation
        self._score += reward
        self._done = terminated or truncated
        return reward, self._done

    def score(self) -> int:
        return self._score

    def is_done(self) -> bool:
        return self._done

    def action_mask(self) -> list[bool]:
        return [True, True, False, False, False]

    def obs_numpy(self) -> np.ndarray:
        return self._obs

    def print_game_state(self) -> None:
        cart_pos, cart_vel, pole_ang, pole_ang_vel = self.obs_numpy()
        print("CartPole:")
        print(f"    Cart pos:   {cart_pos}")
        print(f"    Cart vel:   {cart_vel}")
        print(f"    Pole angle: {pole_ang} rad")
        print(f"    Pole a.v.:  {pole_ang_vel}")
