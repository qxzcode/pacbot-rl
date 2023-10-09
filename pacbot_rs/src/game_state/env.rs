use itertools::izip;
use ndarray::{s, Array, Array3};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use numpy::{IntoPyArray, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::seq::SliceRandom;

use crate::{
    grid::{self, coords_to_node, NODE_COORDS, VALID_ACTIONS},
    variables::{self, GridValue},
};

use super::GameState;

/// How many ticks the game should move every step. Ghosts move every 12 ticks.
const TICKS_PER_STEP: u32 = 8;

/// Whether to randomize the ghosts' positions when `random_start = true`.
const RANDOMIZE_GHOSTS: bool = true;

#[derive(Clone, Copy, Debug, Eq, PartialEq, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
pub enum Action {
    Stay = 0,
    Down = 1,
    Up = 2,
    Left = 3,
    Right = 4,
}

impl<'source> FromPyObject<'source> for Action {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let index: u8 = ob.extract()?;
        Action::try_from_primitive(index).map_err(|_| PyValueError::new_err("Invalid action"))
    }
}

impl IntoPy<PyObject> for Action {
    fn into_py(self, py: Python<'_>) -> PyObject {
        u8::from(self).into_py(py)
    }
}

#[derive(Clone)]
#[pyclass]
pub struct PacmanGym {
    #[pyo3(get)]
    pub game_state: GameState,
    #[pyo3(get, set)]
    pub random_start: bool,
    last_score: u32,
    last_action: Action,
    last_ghost_pos: [(usize, usize); 4],
    last_pos: (usize, usize),
}

#[pymethods]
impl PacmanGym {
    #[new]
    pub fn new(random_start: bool) -> Self {
        let game_state = GameState::new();
        let last_ghost_pos = [
            game_state.red.borrow().current_pos,
            game_state.pink.borrow().current_pos,
            game_state.orange.borrow().current_pos,
            game_state.blue.borrow().current_pos,
        ];
        let mut env = Self {
            random_start,
            last_score: 0,
            last_action: Action::Stay,
            last_ghost_pos,
            last_pos: game_state.pacbot.pos,
            game_state,
        };
        if random_start && RANDOMIZE_GHOSTS {
            for mut ghost in env.game_state.ghosts_mut() {
                ghost.start_path = &[];
            }
        }
        env
    }

    pub fn reset(&mut self) {
        self.last_score = 0;
        self.game_state.restart();

        if self.random_start {
            let rng = &mut rand::thread_rng();
            let mut random_pos = || *NODE_COORDS.choose(rng).unwrap();

            self.game_state.pacbot.update(random_pos());

            if RANDOMIZE_GHOSTS {
                for mut ghost in self.game_state.ghosts_mut() {
                    ghost.current_pos = random_pos();
                    ghost.next_pos = ghost.current_pos;
                }
            }
        }

        self.last_ghost_pos = [
            self.game_state.red.borrow().current_pos,
            self.game_state.pink.borrow().current_pos,
            self.game_state.orange.borrow().current_pos,
            self.game_state.blue.borrow().current_pos,
        ];
        self.last_action = Action::Stay;
        self.last_pos = self.game_state.pacbot.pos;

        self.game_state.unpause();
    }

    /// Performs an action and steps the environment.
    /// Returns (reward, done).
    pub fn step(&mut self, action: Action) -> (i32, bool) {
        // update Pacman pos
        self.last_pos = self.game_state.pacbot.pos;
        self.move_one_cell(action);

        let entity_positions = [
            self.game_state.red.borrow().current_pos,
            self.game_state.pink.borrow().current_pos,
            self.game_state.orange.borrow().current_pos,
            self.game_state.blue.borrow().current_pos,
        ];

        // step through environment multiple times
        // If changing directions, double the number of ticks
        let tick_mult = if self.last_action == action || self.last_action == Action::Stay {
            1
        } else {
            2
        };
        for _ in 0..TICKS_PER_STEP * tick_mult {
            self.game_state.next_step();
            if self.is_done() {
                break;
            }
        }
        self.last_action = action;

        // If the ghost positions change, update the last ghost positions
        let new_entity_positions = [
            self.game_state.red.borrow().current_pos,
            self.game_state.pink.borrow().current_pos,
            self.game_state.orange.borrow().current_pos,
            self.game_state.blue.borrow().current_pos,
        ];
        let pos_changed = entity_positions
            .iter()
            .zip(&new_entity_positions)
            .any(|(e1, e2)| e1 != e2);
        if pos_changed {
            self.last_ghost_pos = entity_positions;
        }

        let done = self.is_done();

        // reward is raw difference in game score, or -100 if eaten
        let mut reward = if done {
            if self.game_state.lives < variables::STARTING_LIVES {
                -200
            } else {
                1000
            }
        } else {
            self.game_state.score as i32 - self.last_score as i32
        };
        if tick_mult == 2 {
            reward -= 10;
        }
        self.last_score = self.game_state.score;

        (reward, done)
    }

    pub fn score(&self) -> u32 {
        self.game_state.score
    }

    pub fn is_done(&self) -> bool {
        !self.game_state.play
    }

    /// Returns the action mask that is `True` for currently-valid actions and
    /// `False` for currently-invalid actions.
    pub fn action_mask(&self) -> [bool; 5] {
        let pacbot_pos = self.game_state.pacbot.pos;
        let pacbot_node = coords_to_node(pacbot_pos).expect("PacBot is in an invalid location");
        VALID_ACTIONS[pacbot_node]
    }

    /// Returns an observation array/tensor constructed from the game state.
    pub fn obs_numpy(&self, py: Python<'_>) -> Py<PyArray3<f32>> {
        self.obs().into_pyarray(py).into()
    }
}

impl PacmanGym {
    fn move_one_cell(&mut self, action: Action) {
        use std::cmp::{max, min};
        let old_pos = self.game_state.pacbot.pos;
        let new_pos = match action {
            Action::Stay => (old_pos.0, old_pos.1),
            Action::Down => (old_pos.0, min(old_pos.1 + 1, grid::GRID[0].len() - 1)),
            Action::Up => (old_pos.0, max(old_pos.1 - 1, 0)),
            Action::Left => (max(old_pos.0 - 1, 0), old_pos.1),
            Action::Right => (min(old_pos.0 + 1, grid::GRID.len() - 1), old_pos.1),
        };
        if grid::is_walkable(new_pos) {
            self.game_state.pacbot.update(new_pos);
        }
    }

    /// Returns an observation array/tensor constructed from the game state.
    pub fn obs(&self) -> Array3<f32> {
        let mut obs_array = Array::zeros((15, 28, 31));
        let (mut wall, mut reward, mut pacman, mut ghost, mut last_ghost, mut state) = obs_array
            .multi_slice_mut((
                s![0, .., ..],
                s![1, .., ..],
                s![2..4, .., ..],
                s![4..8, .., ..],
                s![8..12, .., ..],
                s![12..15, .., ..],
            ));

        let ghost_positions = [
            self.game_state.red.borrow().current_pos,
            self.game_state.pink.borrow().current_pos,
            self.game_state.orange.borrow().current_pos,
            self.game_state.blue.borrow().current_pos,
        ];

        for (grid_value, wall_value, reward_value) in izip!(
            self.game_state.grid.iter().flatten(),
            wall.iter_mut(),
            reward.iter_mut(),
        ) {
            *wall_value = (*grid_value == GridValue::I || *grid_value == GridValue::n) as u8 as f32;
            *reward_value = match grid_value {
                GridValue::o => variables::PELLET_SCORE,
                GridValue::O => variables::POWER_PELLET_SCORE,
                GridValue::c => variables::CHERRY_SCORE,
                _ => 0,
            } as f32
                / variables::GHOST_SCORE as f32;
        }
        if self.game_state.is_frightened() {
            for pos in ghost_positions {
                reward[(pos.0, pos.1)] += 1.0;
            }
        }

        let pac_pos = self.game_state.pacbot.pos;
        pacman[(0, self.last_pos.0, self.last_pos.1)] = 1.0;
        pacman[(1, pac_pos.0, pac_pos.1)] = 1.0;

        let state_index = (self.game_state.state() - 1) as usize;
        for (i, pos) in ghost_positions.iter().enumerate() {
            ghost[(i, pos.0, pos.1)] = 1.0;
            state[(state_index, pos.0, pos.1)] = if state_index == 2 {
                self.game_state.frightened_counter() as f32 / variables::FRIGHTENED_LENGTH as f32
            } else {
                1.0
            };
        }

        for (i, pos) in self.last_ghost_pos.iter().enumerate() {
            last_ghost[(i, pos.0, pos.1)] = 1.0;
        }

        obs_array
    }

    /// Returns an observation array/tensor constructed from the game state.
    pub fn obs_tch(&self) -> tch::Tensor {
        let obs_array = self.obs();
        let shape: Vec<_> = obs_array.shape().iter().map(|&d| d as i64).collect();
        tch::Tensor::from_slice(obs_array.as_slice().unwrap()).reshape(shape)
    }
}
