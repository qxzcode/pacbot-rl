//! AlphaZero-specific utilities.

use numpy::PyArray3;
use pyo3::prelude::*;

use crate::{game_state::env::PacmanGym, mcts::MCTSContext};

#[derive(Clone, Copy, Debug)]
#[pyclass(get_all, set_all)]
pub struct AlphaZeroConfig {
    pub tree_size: usize,
    pub max_episode_length: usize,
    pub discount_factor: f32,
}

#[pymethods]
impl AlphaZeroConfig {
    #[new]
    pub fn new(tree_size: usize, max_episode_length: usize, discount_factor: f32) -> Self {
        Self { tree_size, max_episode_length, discount_factor }
    }

    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[pyclass(get_all)]
pub struct ExperienceItem {
    pub obs: Py<PyArray3<f32>>,
    pub action_mask: [bool; 5],
    pub value: f32,
    pub action_distribution: [f32; 5],
}

#[pyclass]
pub struct ExperienceCollector {
    #[pyo3(get)]
    config: AlphaZeroConfig,
    mcts_context: MCTSContext,
}

#[pymethods]
impl ExperienceCollector {
    #[new]
    pub fn new(evaluator: PyObject, config: AlphaZeroConfig) -> Self {
        let mut env = PacmanGym::new(true);
        env.reset();
        Self { mcts_context: MCTSContext::new(env, evaluator), config }
    }

    /// Generates and returns at least one episode of experience.
    pub fn generate_experience(&mut self, py: Python<'_>) -> Vec<ExperienceItem> {
        assert!(!self.mcts_context.env.is_done());

        let mut experience = Vec::new();

        // Play out an episode, recording the transitions.
        for step_num in 1.. {
            // Get the observation and action mask for the current state.
            let obs = self.mcts_context.root_obs_numpy(py);
            let action_mask = self.mcts_context.env.action_mask();

            // Use MCTS to choose an action.
            let action = self.mcts_context.ponder_and_choose(self.config.tree_size);
            let action_distribution = self.mcts_context.action_distribution();

            // Step the environment and get the reward.
            let (reward, done) = self.mcts_context.take_action(action);

            // Add the transition to the experience buffer.
            experience.push(ExperienceItem {
                obs,
                action_mask,
                value: reward as f32, // This will be updated below to incorporate future steps.
                action_distribution,
            });

            // If the episode terminated, stop.
            if done {
                break;
            }

            // If we've reached the maximum allowed episode length, truncate the episode.
            if step_num >= self.config.max_episode_length {
                // Bootstrap the remaining return from the current value estimate.
                experience.last_mut().unwrap().value +=
                    self.config.discount_factor * self.mcts_context.value();
                break;
            }
        }

        // Backpropagate the rewards to get discounted returns/values.
        let mut next_state_value = 0.0;
        for transition in experience.iter_mut().rev() {
            transition.value += self.config.discount_factor * next_state_value;
            next_state_value = transition.value;
        }

        // Reset the environment.
        self.mcts_context.reset();

        experience
    }
}
