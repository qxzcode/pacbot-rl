This repository contains reinforcement learning implementations for creating agents to control a robot in the Harvard PacBot competition.

I implemented three RL algorithms from scratch: **DQN**, **PPO**, and **AlphaZero**. The main training/evaluation scripts are in `src/algorithms`.

The rest of the `src/` directory contains supporting Python code.

The game environment, as well as some other performance-sentitive code such as the vectorized Monte Carlo tree search implementation for AlphaZero, are implemented in Rust as a Python extension module (using `pyo3` and `maturin`). This code is located in `pacbot_rs/`.
