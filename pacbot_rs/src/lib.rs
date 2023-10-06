pub mod game_state;
pub mod ghost_agent;
pub mod ghost_paths;
pub mod grid;
pub mod heuristic_values;
pub mod mcts;
pub mod observations;
pub mod pacbot;
pub mod variables;

use pyo3::prelude::*;

use mcts::MCTSContext;

use game_state::{env::PacmanGym, GameState};

/// A Python module containing Rust implementations of the PacBot environment.
#[pymodule]
fn pacbot_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GameState>()?;
    m.add_class::<PacmanGym>()?;
    m.add_class::<MCTSContext>()?;
    m.add_function(wrap_pyfunction!(observations::create_obs_semantic, m)?)?;
    m.add_function(wrap_pyfunction!(heuristic_values::get_heuristic_value, m)?)?;
    m.add_function(wrap_pyfunction!(
        heuristic_values::get_action_heuristic_values,
        m
    )?)?;
    Ok(())
}
