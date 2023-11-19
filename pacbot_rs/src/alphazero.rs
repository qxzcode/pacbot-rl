//! AlphaZero-specific utilities.

use itertools::{izip, Itertools};
use ndarray::prelude::*;
use numpy::PyArray3;
use pyo3::prelude::*;

use crate::{
    game_state::env::PacmanGym,
    mcts::{eval_obs_batch, LeafEvaluation, MCTSContext},
};

#[derive(Clone, Copy, Debug)]
#[pyclass(get_all, set_all)]
pub struct AlphaZeroConfig {
    pub tree_size: usize,
    pub max_episode_length: usize,
    pub discount_factor: f32,
    pub num_parallel_envs: usize,
}

#[pymethods]
impl AlphaZeroConfig {
    #[new]
    pub fn new(
        tree_size: usize,
        max_episode_length: usize,
        discount_factor: f32,
        num_parallel_envs: usize,
    ) -> Self {
        assert!(tree_size >= 2);
        assert!(max_episode_length >= 1);
        assert!((0.0..=1.0).contains(&discount_factor));
        assert!(num_parallel_envs >= 1);
        Self { tree_size, max_episode_length, discount_factor, num_parallel_envs }
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

/// All the state for a single parallel MCTS environment.
struct ParallelEnv {
    mcts_context: MCTSContext,
    num_search_iters_remaining: usize,
    experience: Vec<ExperienceItem>,
}

#[pyclass]
pub struct ExperienceCollector {
    #[pyo3(get)]
    config: AlphaZeroConfig,
    evaluator: PyObject,
    mcts_envs: Vec<ParallelEnv>,
}

#[pymethods]
impl ExperienceCollector {
    #[new]
    pub fn new(evaluator: PyObject, config: AlphaZeroConfig) -> PyResult<Self> {
        let mcts_envs = (0..config.num_parallel_envs)
            .map(|_| {
                let mut env = PacmanGym::new(true);
                env.reset();
                Ok(ParallelEnv {
                    mcts_context: MCTSContext::new(env, evaluator.clone())?,
                    num_search_iters_remaining: config.tree_size - 1,
                    experience: Vec::new(),
                })
            })
            .collect::<PyResult<_>>()?;
        Ok(Self { config, evaluator, mcts_envs })
    }

    /// Generates and returns at least one episode of experience.
    pub fn generate_experience(&mut self) -> PyResult<Vec<ExperienceItem>> {
        let mut final_experience = Vec::new();
        while final_experience.is_empty() {
            self.generate_experience_step(&mut final_experience)?;
        }
        Ok(final_experience)
    }
}

impl ExperienceCollector {
    /// Performs one parallel MCTS search+eval iteration, across all the parallel environments.
    /// Appends any experience from completed episodes into the provided `Vec`.
    fn generate_experience_step(
        &mut self,
        final_experience: &mut Vec<ExperienceItem>,
    ) -> PyResult<()> {
        assert!(!itertools::any(&self.mcts_envs, |env| env.mcts_context.env.is_done()));

        // Gather a batch of observations from all the parallel environments.
        let (trajectories, observations): (Vec<_>, Vec<_>) =
            self.mcts_envs.iter().map(|env| env.mcts_context.select_trajectory()).unzip();

        // Store which envs need obs evaluation.
        let env_mask = observations.iter().map(|obs| obs.is_some());
        let observations = observations.iter().filter_map(|obs| obs.as_ref());

        // Stack the observations into arrays.
        macro_rules! stack {
            ($iter:expr) => {
                ndarray::stack(Axis(0), &$iter.collect_vec()).unwrap()
            };
        }
        let action_masks = stack!(observations.clone().map(|(_, mask)| ArrayView1::from(mask)));
        let obs = stack!(observations.map(|(obs, _)| obs.view()));

        // Evaluate the batch.
        let mut leaf_evaluations = Python::with_gil(|py| -> PyResult<_> {
            let (values, policies) = eval_obs_batch(py, obs, action_masks, &self.evaluator)?;

            // Zip the results into an iterator of LeafEvaluation objects.
            let policies = policies.as_array();
            Ok(izip!(values.as_array(), policies.outer_iter())
                .map(|(&value, policy)| LeafEvaluation {
                    value,
                    policy: array_init::from_iter(policy.iter().copied()).unwrap(),
                })
                .collect_vec()
                .into_iter())
        })?;

        // Backpropagate all the environments' trajectories, distributing the evaluations as needed.
        // Also collect experience from any completed episodes.
        for (env, trajectory, needs_eval) in izip!(&mut self.mcts_envs, trajectories, env_mask) {
            // Backpropagate the trajectory.
            let leaf_eval = needs_eval.then(|| leaf_evaluations.next().unwrap());
            env.mcts_context.backprop_trajectory(&trajectory, leaf_eval);

            // Check if this env has done enough search iterations to take an action.
            env.num_search_iters_remaining -= 1;
            if env.num_search_iters_remaining == 0 {
                // Get the observation, action mask, action distribution, and best action.
                let obs = Python::with_gil(|py| env.mcts_context.root_obs_numpy(py));
                let action_mask = env.mcts_context.env.action_mask();
                let action_distribution = env.mcts_context.action_distribution();
                let action = env.mcts_context.best_action();

                // Step the environment and get the reward.
                let (reward, done) = env.mcts_context.take_action(action)?;

                // Add the transition to the env's local experience buffer.
                env.experience.push(ExperienceItem {
                    obs,
                    action_mask,
                    value: reward as f32, // This will be updated later to incorporate future steps.
                    action_distribution,
                });

                // Helper function that finalizes the experience steps and resets the environment.
                let mut end_episode = |env: &mut ParallelEnv| {
                    // Backpropagate the rewards to get discounted returns/values.
                    let mut next_state_value = 0.0;
                    for transition in env.experience.iter_mut().rev() {
                        transition.value += self.config.discount_factor * next_state_value;
                        next_state_value = transition.value;
                    }

                    // Move the finalized experience steps to the output buffer.
                    final_experience.append(&mut env.experience);

                    // Reset the environment and search context.
                    env.mcts_context.reset()
                };

                if done {
                    // The episode terminated.
                    end_episode(env)?;
                } else if env.experience.len() >= self.config.max_episode_length {
                    // We've reached the maximum allowed episode length; truncate the episode.
                    // Bootstrap the remaining return from the current value estimate.
                    env.experience.last_mut().unwrap().value +=
                        self.config.discount_factor * env.mcts_context.value();
                    end_episode(env)?;
                }

                // Reset the number of search iters for the next action.
                env.num_search_iters_remaining =
                    self.config.tree_size.saturating_sub(env.mcts_context.node_count());
            }
        }

        Ok(())
    }
}
