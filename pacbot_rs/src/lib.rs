pub mod alphazero;
pub mod env;
pub mod grid;
pub mod mcts;

use pyo3::prelude::*;

use alphazero::{AlphaZeroConfig, ExperienceCollector, ExperienceItem};
use env::PacmanGym;
use mcts::MCTSContext;

/// A Python module containing Rust implementations of the PacBot environment.
#[pymodule]
fn pacbot_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PacmanGym>()?;
    m.add_class::<MCTSContext>()?;
    m.add_class::<AlphaZeroConfig>()?;
    m.add_class::<ExperienceItem>()?;
    m.add_class::<ExperienceCollector>()?;
    Ok(())
}
