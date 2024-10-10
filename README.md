# What is this repo?

This repository contains reinforcement learning implementations for creating agents to control a robot in the Harvard PacBot competition.

I implemented three RL algorithms from scratch: **DQN**, **PPO**, and **AlphaZero**. The main training/evaluation scripts are in `src/algorithms/`.

The rest of the `src/` directory contains supporting Python code.

The game environment, as well as some other performance-sentitive code such as the vectorized Monte Carlo tree search implementation for AlphaZero, are implemented in Rust as a Python extension module (using `pyo3` and `maturin`). This code is located in `pacbot_rs/`.


# Usage

Make sure you have [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) and [Rust](https://www.rust-lang.org/tools/install) installed.

DQN has generally worked better than PPO and AlphaZero for us, so the main script you should run is `src/algorithms/train_dqn.py`.

## Setup

Make sure you have the poetry environment activated (by running `poetry shell` within the repo; make sure you have also installed dependencies with `poetry install`).

Initially, and after making changes to any of the Rust code in `pacbot_rs/`, you'll need to build the Rust extension module and install it into the Python environment:
```bash
pacbot_rs/build_and_install.sh
```

## Training

First, make sure your working directory is `src/`.

To start a training run, invoke `train_dqn.py`:
```bash
python3 -m algorithms.train_dqn
```
 - To fine-tune (start from an existing checkpoint instead of initializing the model randomly), pass `--finetune path/to/checkpoint`.
 - By default, model checkpoints will be saved to `./checkpoints/`. You can change this by passing `--checkpoint-dir path/to/dir`.
 - By default, this will use [Weights & Biases](https://wandb.ai/) to log checkpoints and various metrics to aid debugging and tracking training progress. To disable this, pass `--no-wandb`.
 - By default, the `cuda` device is used (if available), falling back to `cpu`. To manually set the PyTorch device, pass `--device <device>`.

You can also change **hyperparameters** with command-line arguments. To see the full list, pass `--help` or take a look at the `hyperparam_defaults` dictionary near the top of `train_dqn.py`. In particular:
 - If you're running out of memory, consider reducing the `batch_size`.
 - `num_iters` controls the total runtime of the script execution, and also indirectly the value of $\epsilon$ (epsilon), which controls the amount of random exploration ($\epsilon$ starts at `initial_epsilon` and linearly decreases to `final_epsilon` after `num_iters`). Of course, you can always set `num_iters` to a large value (like the default) and just kill the script once it reaches a satisfactory level of performance.

## Visualizing a trained agent

To show an animated visualization in the terminal of the agent being controlled by a particular checkpoint, use `--eval`:
```bash
python3 -m algorithms.train_dqn --eval path/to/checkpoint.pt
```
