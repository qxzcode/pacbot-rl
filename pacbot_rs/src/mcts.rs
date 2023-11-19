use ndarray::Array3;
use num_enum::TryFromPrimitive;
use numpy::IntoPyArray;
use ordered_float::NotNan;
use pyo3::prelude::*;

use crate::game_state::env::{Action, PacmanGym};

/// The type for returns (cumulative rewards).
type Return = f32;

/// The factor used to discount future rewards each timestep.
const DISCOUNT_FACTOR: f32 = 0.99;

struct SearchTreeEdge {
    policy_prior: f32,
    visit_count: u32,
    total_return: Return,
    child: Option<Box<SearchTreeNode>>,
}

impl SearchTreeEdge {
    /// Creates a new edge with the given policy prior probability.
    #[must_use]
    fn new(policy_prior: f32) -> Self {
        Self { policy_prior, visit_count: 0, total_return: 0.0, child: None }
    }

    /// Returns the estimated expected return (cumulative reward) for this action.
    #[must_use]
    fn expected_return(&self) -> NotNan<f32> {
        if self.visit_count == 0 {
            NotNan::new(0.0).unwrap() // TODO: handle this case in a more principled way?
        } else {
            let expected_return = self.total_return / (self.visit_count as f32);
            NotNan::new(expected_return).expect("expected score is NaN")
        }
    }

    /// A variant of the PUCT score, similar to that used in AlphaZero.
    #[must_use]
    fn puct_score(&self, parent_visit_count: u32) -> NotNan<f32> {
        let exploration_rate = 100.0; // TODO: make this a tunable parameter
        let exploration_score = self.policy_prior
            * ((parent_visit_count as f32).sqrt() / ((1 + self.visit_count) as f32));
        self.expected_return() + exploration_rate * exploration_score
    }
}

struct SearchTreeNode {
    visit_count: u32,
    children: [SearchTreeEdge; 5],
}

impl SearchTreeNode {
    /// Creates a new node with the given policy prior probabilities.
    #[must_use]
    fn new(policy_priors: [f32; 5]) -> Self {
        Self { visit_count: 0, children: policy_priors.map(SearchTreeEdge::new) }
    }

    /// Returns the index of the next action to sample from the state, using the PUCT criterion.
    #[must_use]
    fn pick_action(&self, env: &PacmanGym) -> usize {
        // Choose a (valid) action based on the current stats.
        let action_mask = env.action_mask();
        let (action_index, _) = self
            .children
            .iter()
            .enumerate()
            .filter(|&(action_index, _)| action_mask[action_index])
            .max_by_key(|(_, edge)| edge.puct_score(self.visit_count))
            .unwrap();
        action_index
    }

    /// Returns the valid action with the highest expected return.
    #[must_use]
    fn best_action(&self, env: &PacmanGym) -> Action {
        let action_mask = env.action_mask();
        let (action_index, _) = self
            .children
            .iter()
            .enumerate()
            .filter(|&(action_index, _)| action_mask[action_index])
            .max_by_key(|(_, edge)| edge.expected_return())
            .unwrap();
        Action::try_from_primitive(action_index.try_into().unwrap()).unwrap()
    }

    /// Returns the maximum node depth in this subtree.
    #[must_use]
    fn max_depth(&self) -> usize {
        self.children
            .iter()
            .filter_map(|edge| edge.child.as_ref())
            .map(|child| 1 + child.max_depth())
            .max()
            .unwrap_or(0)
    }

    /// Returns the total number of nodes in this subtree, including this node.
    #[must_use]
    fn node_count(&self) -> usize {
        1 + self
            .children
            .iter()
            .filter_map(|edge| edge.child.as_ref())
            .map(|child| child.node_count())
            .sum::<usize>()
    }
}

#[pyclass]
pub struct MCTSContext {
    #[pyo3(get)]
    env: PacmanGym,

    root: SearchTreeNode,

    #[pyo3(get, set)]
    evaluator: PyObject,
}

#[pymethods]
impl MCTSContext {
    /// Creates a new MCTS context with the given environment and evaluator.
    #[new]
    pub fn new(env: PacmanGym, evaluator: PyObject) -> Self {
        let root_evaluation = eval_obs(env.obs(), &evaluator);
        Self { env, root: SearchTreeNode::new(root_evaluation.policy), evaluator }
    }

    /// Resets the environment and search tree, starting a new episode.
    pub fn reset(&mut self) {
        // Reset the environment.
        self.env.reset();

        // Reset the search tree.
        let root_evaluation = eval_obs(self.env.obs(), &self.evaluator);
        self.root = SearchTreeNode::new(root_evaluation.policy);
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
                .child
                .take()
                .map(|child_box| *child_box)
                .unwrap_or_else(|| {
                    let root_evaluation = eval_obs(self.env.obs(), &self.evaluator);
                    SearchTreeNode::new(root_evaluation.policy)
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
    #[must_use]
    pub fn action_distribution(&self) -> [f32; 5] {
        array_init::map_array_init(&self.root.children, |edge| {
            edge.visit_count as f32 / self.root.visit_count as f32
        })
    }

    /// Returns the estimated action (Q) values at the root node.
    #[must_use]
    pub fn action_values(&self) -> [f32; 5] {
        array_init::map_array_init(&self.root.children, |edge| edge.expected_return().into())
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
}

struct Transition {
    action_index: usize,
    reward: i32,
}

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
            } else if let Some(child_node) = &node.children[action_index].child {
                // Update the current node and continue.
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
            let edge = &mut node.children[transition.action_index];

            // (Recursively) get the subsequent return, then compute the return for this node.
            let subsequent_return = {
                if rest.is_empty() {
                    // The trajectory ends here.
                    if let Some(LeafEvaluation { value, policy }) = leaf_evaluation {
                        // We have a leaf evaluation, so expand a new leaf node.
                        edge.child = Some(Box::new(SearchTreeNode::new(policy)));
                        value
                    } else {
                        // The trajectory reached a terminal node, so there is no child node and the
                        // return from here is zero.
                        0.0
                    }
                } else {
                    let child = edge.child.as_mut().expect("backprop hit an unexpanded child");
                    backprop_helper(child, rest, leaf_evaluation)
                }
            };
            let this_return = transition.reward as Return + DISCOUNT_FACTOR * subsequent_return;

            // Update this node/edge.
            node.visit_count += 1;
            edge.visit_count += 1;
            edge.total_return += this_return;

            // Propagate the return up the trajectory.
            this_return
        }

        backprop_helper(&mut self.root, trajectory, leaf_evaluation);
    }
}

/// Evaluates the given observation using the given neural evaluation function.
#[must_use]
fn eval_obs(obs: Array3<f32>, evaluator: &PyObject) -> LeafEvaluation {
    Python::with_gil(|py| {
        // Convert obs into a NumPy array.
        let obs_numpy = obs.into_pyarray(py);

        // Run the evaluator.
        let result = evaluator.call1(py, (obs_numpy,)).expect("PyTorch error");

        // Convert the result to Rust types.
        let (value, policy) = result.extract(py).expect("evaluator should return (value, policy)");
        LeafEvaluation { value, policy }
    })
}
