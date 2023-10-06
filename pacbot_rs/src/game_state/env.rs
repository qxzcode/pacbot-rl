use num_enum::{IntoPrimitive, TryFromPrimitive};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use tch::IndexOp;

use crate::{
    grid::{self, coords_to_node, NODE_COORDS, VALID_ACTIONS},
    variables,
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
    last_ghost_pos: Vec<(usize, usize)>,
    last_pos: (usize, usize),
}

#[pymethods]
impl PacmanGym {
    #[new]
    pub fn new(random_start: bool) -> Self {
        let game_state = GameState::new();
        let mut env = Self {
            random_start,
            last_score: 0,
            last_action: Action::Stay,
            last_ghost_pos: vec![
                game_state.red.borrow().current_pos,
                game_state.pink.borrow().current_pos,
                game_state.orange.borrow().current_pos,
                game_state.blue.borrow().current_pos,
            ],
            last_pos: game_state.pacbot.pos,
            game_state: game_state.clone(),
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

        self.last_ghost_pos = vec![
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
            self.last_ghost_pos.copy_from_slice(&entity_positions);
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

    /// Returns an observation for the value network.
    pub fn obs(&self) -> tch::Tensor {
        let grid_vec: Vec<u8> = self
            .game_state
            .grid
            .iter()
            .flatten()
            .map(|cell| *cell as u8)
            .collect();
        let grid_width = 28;
        let grid_height = 31;
        let wall_vec: Vec<u8> = grid_vec
            .iter()
            .map(|&cell| (cell == 1 || cell == 5) as u8)
            .collect();
        let wall = tch::Tensor::of_slice(&wall_vec).reshape(&[grid_width, grid_height]);

        let fright = self.game_state.is_frightened();
        let entity_positions = [
            self.game_state.red.borrow().current_pos,
            self.game_state.pink.borrow().current_pos,
            self.game_state.orange.borrow().current_pos,
            self.game_state.blue.borrow().current_pos,
        ];
        let ghost = tch::Tensor::zeros(
            &[4, grid_width, grid_height],
            (tch::Kind::Int, tch::Device::Cpu),
        );
        let state = tch::Tensor::zeros(
            &[3, grid_width, grid_height],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        let fright_ghost = tch::Tensor::zeros(
            &[grid_width, grid_height],
            (tch::Kind::Float, tch::Device::Cpu),
        );
        for (i, pos) in entity_positions.iter().enumerate() {
            ghost.i((i as i64, pos.0 as i64, pos.1 as i64)).fill_(1);
            fright_ghost
                .i((pos.0 as i64, pos.1 as i64))
                .fill_(fright as i64);
            state
                .i((
                    (self.game_state.state() as u8 - 1) as i64,
                    pos.0 as i64,
                    pos.1 as i64,
                ))
                .fill_(1.0);
            if i == 3 {
                state
                    .i((
                        (self.game_state.state() as u8 - 1) as i64,
                        pos.0 as i64,
                        pos.1 as i64,
                    ))
                    .fill_(
                        self.game_state.frightened_counter() as f64
                            / variables::FRIGHTENED_LENGTH as f64,
                    );
            }
        }

        let last_ghost = tch::Tensor::zeros(&ghost.size(), (tch::Kind::Int, tch::Device::Cpu));
        for (i, pos) in self.last_ghost_pos.iter().enumerate() {
            last_ghost
                .i((i as i64, pos.0 as i64, pos.1 as i64))
                .fill_(1);
        }

        let reward_vec: Vec<f32> = grid_vec.iter().map(|cell| match cell {
            2 => variables::PELLET_SCORE,
            6 => variables::CHERRY_SCORE,
            4 => variables::POWER_PELLET_SCORE,
            _ => 0
        } as f32 / variables::GHOST_SCORE as f32).collect();
        let reward =
            tch::Tensor::of_slice(&reward_vec).reshape(&[grid_width, grid_height]) + fright_ghost;

        let pac_pos = self.game_state.pacbot.pos;
        let pacman = tch::Tensor::zeros(
            &[2, grid_width, grid_height],
            (tch::Kind::Int, tch::Device::Cpu),
        );
        pacman
            .i((0, self.last_pos.0 as i64, self.last_pos.1 as i64))
            .fill_(1);
        pacman.i((1, pac_pos.0 as i64, pac_pos.1 as i64)).fill_(1);
        tch::Tensor::concat(
            &[
                tch::Tensor::stack(&[wall, reward], 0),
                pacman,
                ghost,
                last_ghost,
                state,
            ],
            0,
        )
    }
}
