use itertools::izip;
use ndarray::Array3;
use num_enum::TryFromPrimitive;
use numpy::{IntoPyArray, PyArray3};
use ordered_float::NotNan;
use pyo3::prelude::*;

use crate::game_state::env::{Action, PacmanGym};

/// The type for returns (cumulative rewards).
type Return = f32;

/// The factor used to discount future rewards each timestep.
pub const DISCOUNT_FACTOR: f32 = 0.99;

#[derive(Debug)]
struct SearchTreeNode {
    visit_count: u32,
    total_return: Return,
    policy_priors: [f32; 5],
    children: [Option<Box<SearchTreeNode>>; 5],
}

impl SearchTreeNode {
    /// Creates a new node with the given policy prior probabilities and a visit_count of 1.
    #[must_use]
    fn new_visited(initial_evaluation: LeafEvaluation) -> Self {
        Self {
            visit_count: 1,
            total_return: initial_evaluation.value,
            policy_priors: initial_evaluation.policy,
            children: Default::default(),
        }
    }

    /// Creates a new unvisited node with the given policy prior probabilities.
    #[must_use]
    fn new(policy_priors: [f32; 5]) -> Self {
        Self { visit_count: 0, total_return: 0.0, policy_priors, children: Default::default() }
    }

    /// Creates a new, unvisited terminal node.
    #[must_use]
    fn new_terminal() -> Self {
        Self::new([0.0; 5])
    }

    /// Returns the index of the next action to sample from the state, using the PUCT criterion.
    #[must_use]
    fn pick_action(&self, env: &PacmanGym) -> usize {
        // Choose a (valid) action based on the current stats.
        let (action_index, _) = izip!(env.action_mask(), &self.children, self.policy_priors)
            .enumerate()
            .filter(|&(_, (is_valid, _, _))| is_valid)
            .max_by_key(|&(_, (_, child, prior))| {
                // Compute the PUCT score for this child.
                let exploration_rate = 200.0; // TODO: make this a tunable parameter

                let child_visit_count = child.as_ref().map_or(0, |child| child.visit_count);
                let child_value = self.child_value(child);

                let exploration_score =
                    prior * (self.visit_count as f32).sqrt() / ((1 + child_visit_count) as f32);

                child_value + exploration_rate * exploration_score
            })
            .unwrap();
        action_index
    }

    /// Returns the valid action with the highest expected return.
    #[must_use]
    fn best_action(&self, env: &PacmanGym) -> Action {
        let (action_index, _) = izip!(env.action_mask(), &self.children)
            .enumerate()
            .filter(|&(_, (is_valid, _))| is_valid)
            .max_by_key(|(_, (_, child))| self.child_value(child))
            .unwrap();
        Action::try_from_primitive(action_index.try_into().unwrap()).unwrap()
    }

    /// Returns the estimated expected return (cumulative reward) for this node/state.
    #[must_use]
    fn value(&self) -> NotNan<f32> {
        let expected_return = self.total_return / self.visit_count as f32;
        NotNan::new(expected_return).expect("expected return is NaN")
    }

    /// Returns the estimated expected return (cumulative reward) for a child of this node.
    #[must_use]
    fn child_value(&self, child: &Option<Box<SearchTreeNode>>) -> NotNan<f32> {
        child.as_ref().map_or_else(|| self.value(), |child| child.value())
    }

    /// Returns the maximum node depth in this subtree.
    #[must_use]
    fn max_depth(&self) -> usize {
        self.children
            .iter()
            .filter_map(|child| child.as_ref().map(|child| 1 + child.max_depth()))
            .max()
            .unwrap_or(0)
    }

    /// Returns the total number of nodes in this subtree, including this node.
    #[must_use]
    fn node_count(&self) -> usize {
        1 + self
            .children
            .iter()
            .filter_map(|child| child.as_ref().map(|child| child.node_count()))
            .sum::<usize>()
    }
}

#[pyclass]
pub struct MCTSContext {
    #[pyo3(get)]
    pub env: PacmanGym,

    root: SearchTreeNode,

    #[pyo3(get, set)]
    pub evaluator: PyObject,
}

#[pymethods]
impl MCTSContext {
    /// Creates a new MCTS context with the given environment and evaluator.
    #[new]
    pub fn new(env: PacmanGym, evaluator: PyObject) -> Self {
        let root_evaluation = eval_obs(env.obs(), &evaluator);
        Self { env, root: SearchTreeNode::new_visited(root_evaluation), evaluator }
    }

    /// Resets the environment and search tree, starting a new episode.
    pub fn reset(&mut self) {
        // Reset the environment.
        self.env.reset();

        // Reset the search tree.
        let root_evaluation = eval_obs(self.env.obs(), &self.evaluator);
        self.root = SearchTreeNode::new_visited(root_evaluation);
    }

    /// Takes the given action, stepping the environment and updating the search tree root to the
    /// resulting child node.
    ///
    /// Returns (reward, done).
    ///
    /// Panics if `self.env.is_done()`.
    pub fn take_action(&mut self, action: Action) -> (i32, bool) {
        assert!(!self.env.is_done());

        // Step the environment.
        let (reward, done) = self.env.step(action);

        if !done {
            // Update the root node.
            self.root = self.root.children[u8::from(action) as usize]
                .take()
                .map(|child_box| *child_box)
                .unwrap_or_else(|| {
                    let root_evaluation = eval_obs(self.env.obs(), &self.evaluator);
                    SearchTreeNode::new_visited(root_evaluation)
                });
        }

        (reward, done)
    }

    /// Returns the action at the root with the highest expected return.
    #[must_use]
    pub fn best_action(&self) -> Action {
        self.root.best_action(&self.env)
    }

    /// Returns the action visit distribution at the root node.
    ///
    /// Panics if no children have been visited.
    #[must_use]
    pub fn action_distribution(&self) -> [f32; 5] {
        assert!(self.root.visit_count > 1);
        let total_child_visits = (self.root.visit_count - 1) as f32;
        array_init::map_array_init(&self.root.children, |child| {
            child.as_ref().map_or(0.0, |child| child.visit_count as f32 / total_child_visits)
        })
    }

    /// Returns the estimated action (Q) values at the root node.
    #[must_use]
    pub fn action_values(&self) -> [f32; 5] {
        array_init::map_array_init(&self.root.children, |child| self.root.child_value(child).into())
    }

    /// Returns the estimated value/return at the root node.
    #[must_use]
    pub fn value(&self) -> f32 {
        self.root.value().into()
    }

    /// Performs MCTS iterations to grow the tree to (approximately) the given size,
    /// then returns the best action.
    ///
    /// Panics if `self.env.is_done()`.
    pub fn ponder_and_choose(&mut self, max_tree_size: usize) -> Action {
        let num_iterations = max_tree_size.saturating_sub(self.node_count());
        for _ in 0..num_iterations {
            let (trajectory, leaf_obs) = self.select_trajectory();
            let leaf_evaluation = leaf_obs.map(|leaf_obs| eval_obs(leaf_obs, &self.evaluator));
            self.backprop_trajectory(&trajectory, leaf_evaluation);
        }

        self.best_action()
    }

    /// Returns the maximum node depth in the current search tree.
    #[must_use]
    pub fn max_depth(&self) -> usize {
        self.root.max_depth()
    }

    /// Returns the total number of nodes in the current search tree.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.root.node_count()
    }

    /// Returns the observation at the current root node.
    #[must_use]
    pub fn root_obs_numpy(&self, py: Python<'_>) -> Py<PyArray3<f32>> {
        self.env.obs().into_pyarray(py).into()
    }
}

#[derive(Debug)]
struct Transition {
    action_index: usize,
    reward: i32,
}

#[derive(Clone, Copy)]
struct LeafEvaluation {
    value: Return,
    policy: [f32; 5],
}

impl MCTSContext {
    /// Selects a trajectory starting at the current root node/state and ending at either
    /// an unexpanded node or a terminal state.
    ///
    /// If it ended at an unexpanded node, the last observation is returned for evaluation.
    /// Otherwise (if it ended at a terminal state) no observation is returned.
    ///
    /// Panics if `self.env.is_done()`.
    #[must_use]
    fn select_trajectory(&self) -> (Vec<Transition>, Option<Array3<f32>>) {
        assert!(!self.env.is_done());
        let mut env = self.env.clone();

        let mut trajectory = Vec::new();
        let mut node = &self.root;

        loop {
            // Pick an action to traverse.
            let action_index = node.pick_action(&env);

            // Take the action and record the transition.
            let (reward, done) = env.step(Action::from_index(action_index));
            trajectory.push(Transition { action_index, reward });

            if done {
                // We've reached a terminal state; stop.
                return (trajectory, None);
            } else if let Some(child_node) = &node.children[action_index] {
                // The child for the chosen action exists; update the current node and continue.
                node = child_node;
            } else {
                // We've reached an unexpanded node; stop and return the observation.
                return (trajectory, Some(env.obs()));
            }
        }
    }

    /// Expand a new leaf node and update the nodes along a trajectory by backpropagating rewards
    /// from the given leaf return.
    fn backprop_trajectory(
        &mut self,
        trajectory: &[Transition],
        leaf_evaluation: Option<LeafEvaluation>,
    ) {
        fn backprop_helper(
            node: &mut SearchTreeNode,
            trajectory: &[Transition],
            leaf_evaluation: Option<LeafEvaluation>,
        ) -> Return {
            // Get the next action edge.
            let (transition, rest) =
                trajectory.split_first().expect("backprop called with empty trajectory");
            let child = &mut node.children[transition.action_index];

            // Create/get the child node and the subsequent return.
            let (child, subsequent_return) = if rest.is_empty() {
                // The trajectory ends here.
                if let Some(LeafEvaluation { value, policy }) = leaf_evaluation {
                    // We have a leaf evaluation, so expand a new leaf node.
                    assert!(child.is_none(), "tried to expand the same child twice");
                    (child.insert(Box::new(SearchTreeNode::new(policy))), value)
                } else {
                    // The trajectory reached a terminal node, so the return from here is zero.
                    (child.get_or_insert_with(|| Box::new(SearchTreeNode::new_terminal())), 0.0)
                }
            } else {
                // Recursively get the subsequent return from the existing child.
                let child = child.as_mut().expect("backprop hit an unexpanded child");
                let subsequent_return = backprop_helper(child, rest, leaf_evaluation);
                (child, subsequent_return)
            };

            // Update the child node.
            let this_return = transition.reward as Return + DISCOUNT_FACTOR * subsequent_return;
            child.visit_count += 1;
            child.total_return += this_return;

            // Propagate the return back up the trajectory.
            this_return
        }

        let subsequent_return = backprop_helper(&mut self.root, trajectory, leaf_evaluation);
        self.root.visit_count += 1;
        self.root.total_return += DISCOUNT_FACTOR * subsequent_return;
    }
}

/// Evaluates the given observation using the given neural evaluation function.
#[must_use]
fn eval_obs(obs: Array3<f32>, evaluator: &PyObject) -> LeafEvaluation {
    Python::with_gil(|py| {
        // Convert obs into a NumPy array.
        let obs_numpy = obs.into_pyarray(py);

        // Run the evaluator.
        let result = evaluator.call1(py, (obs_numpy,)).expect("error running evaluator");

        // Convert the result to Rust types.
        let (value, policy) = result.extract(py).expect("evaluator should return (value, policy)");
        LeafEvaluation { value, policy }
    })
}
