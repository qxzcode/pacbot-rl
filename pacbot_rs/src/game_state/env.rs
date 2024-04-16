use itertools::izip;
use ndarray::{s, Array, Array3};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use numpy::{IntoPyArray, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{seq::SliceRandom, Rng};

use crate::{
    grid::{self, coords_to_node, NODE_COORDS, VALID_ACTIONS},
    variables::{self, GridValue, CHASE_DURATION, STARTING_LIVES, TICKS_PER_UPDATE},
};

use super::{GameState, GameStateState};

/// How many ticks the game should move every step normally. Ghosts move every 12 ticks.
const NORMAL_TICKS_PER_STEP: u32 = 8;

/// Minimum number of ticks per step.
const MIN_TICKS_PER_STEP: u32 = 4;

/// Maximum number of ticks per step.
const MAX_TICKS_PER_STEP: u32 = 14;

/// Whether to randomize the ghosts' positions when `random_start = true`.
const RANDOMIZE_GHOSTS: bool = true;

/// Penalty for turning.
const TURN_PENALTY: i32 = -10;

#[derive(Clone, Copy, Debug, Eq, PartialEq, TryFromPrimitive, IntoPrimitive)]
#[repr(u8)]
pub enum Action {
    Stay = 0,
    Down = 1,
    Up = 2,
    Left = 3,
    Right = 4,
}

impl Action {
    /// Converts the given action index into an `Action`.
    ///
    /// Panics if index is outside the range `0..5`.
    pub fn from_index(index: usize) -> Self {
        Self::try_from_primitive(index.try_into().unwrap()).unwrap()
    }
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
    ticks_per_step: u32,
    random_ticks: bool,
}

#[pymethods]
impl PacmanGym {
    #[new]
    pub fn new(random_start: bool, random_ticks: bool) -> Self {
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
            ticks_per_step: NORMAL_TICKS_PER_STEP,
            random_ticks: random_ticks,
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
        
        let rng = &mut rand::thread_rng();
        if self.random_ticks {
            self.ticks_per_step = rng.gen_range(MIN_TICKS_PER_STEP..MAX_TICKS_PER_STEP);
        }

        if self.random_start {
            let mut random_pos = || *NODE_COORDS.choose(rng).unwrap();

            self.game_state.pacbot.update(random_pos());

            if RANDOMIZE_GHOSTS {
                for mut ghost in self.game_state.ghosts_mut() {
                    ghost.current_pos = random_pos();
                    ghost.next_pos = ghost.current_pos;
                }
            }

            // Randomly remove pellets from half the board (left, right, top, bottom) or don't.
            let wipe_type = rng.gen_range(0..=4);
            if wipe_type != 0 {
                for (x, col) in self.game_state.grid.iter_mut().enumerate() {
                    for (y, cell) in col.iter_mut().enumerate() {
                        if matches!(*cell, GridValue::o | GridValue::O)
                            && match wipe_type {
                                1 => x < 28 / 2,
                                2 => x >= 28 / 2,
                                3 => y < 31 / 2,
                                4 => y >= 31 / 2,
                                _ => unreachable!(),
                            }
                        {
                            if *cell == GridValue::o {
                                self.game_state.pellets -= 1;
                            } else {
                                self.game_state.power_pellets -= 1;
                            }
                            *cell = GridValue::e;
                        }
                    }
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
        let turn_penalty = if self.last_action == action || self.last_action == Action::Stay { 0 } else { TURN_PENALTY };
        for _ in 0..self.ticks_per_step {
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
        let pos_changed =
            entity_positions.iter().zip(&new_entity_positions).any(|(e1, e2)| e1 != e2);
        if pos_changed {
            self.last_ghost_pos = entity_positions;
        }

        let done = self.is_done();

        // The reward is raw difference in game score, minus a penalty for dying or
        // plus a bonus for clearing the board.
        let mut reward = (self.game_state.score as i32 - self.last_score as i32) + turn_penalty;
        if done {
            if self.game_state.lives < STARTING_LIVES {
                // Pacman died.
                reward += -200;
            } else {
                // Pacman cleared the board! Good Pacman.
                reward += 3_000;
            }
        }
        self.last_score = self.game_state.score;

        (reward, done)
    }

    pub fn score(&self) -> u32 {
        self.game_state.score
    }

    pub fn lives(&self) -> u8 {
        self.game_state.lives
    }

    pub fn is_done(&self) -> bool {
        !self.game_state.play
    }

    pub fn remaining_pellets(&self) -> u32 {
        self.game_state.pellets
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

    /// Prints a representation of the game state to standard output.
    pub fn print_game_state(&self) {
        // Print the score.
        print!("Score: {}", self.score());
        if self.is_done() {
            print!("  [DONE]");
        }
        println!();

        // Print the game grid.
        let ghost_char = |x, y| {
            for (ch, ghost) in [
                ('R', self.game_state.red.borrow()),
                ('P', self.game_state.pink.borrow()),
                ('O', self.game_state.orange.borrow()),
                ('B', self.game_state.blue.borrow()),
            ] {
                if (x, y) == ghost.current_pos {
                    let color = if ghost.is_frightened() { "96" } else { "38;5;206" };
                    return Some((ch, color));
                }
            }
            None
        };
        for y in (0..self.game_state.grid[0].len()).rev() {
            for x in 0..self.game_state.grid.len() {
                let (ch, style) = if (x, y) == self.game_state.pacbot.pos {
                    ('@', "93")
                } else if let Some(ch) = ghost_char(x, y) {
                    ch
                } else {
                    match self.game_state.grid[x][y] {
                        GridValue::I => ('#', "90"),
                        GridValue::o => ('.', ""),
                        GridValue::e => (' ', ""),
                        GridValue::O => ('o', ""),
                        GridValue::n => ('n', "90"),
                        GridValue::c => ('c', "31"),
                    }
                };
                print!("\x1b[{style}m{ch}\x1b[0m");
            }
            println!();
        }
    }
}

impl PacmanGym {
    fn move_one_cell(&mut self, action: Action) {
        use std::cmp::min;
        let old_pos = self.game_state.pacbot.pos;
        let new_pos = match action {
            Action::Stay => (old_pos.0, old_pos.1),
            Action::Down => (old_pos.0, min(old_pos.1 + 1, grid::GRID[0].len() - 1)),
            Action::Up => (old_pos.0, old_pos.1.saturating_sub(1)),
            Action::Left => (old_pos.0.saturating_sub(1), old_pos.1),
            Action::Right => (min(old_pos.0 + 1, grid::GRID.len() - 1), old_pos.1),
        };
        if grid::is_walkable(new_pos) {
            self.game_state.pacbot.update(new_pos);
        }
    }

    /// Returns an observation array/tensor constructed from the game state.
    pub fn obs(&self) -> Array3<f32> {
        let mut obs_array = Array::zeros((16, 28, 31));
        let (mut wall, mut reward, mut pacman, mut ghost, mut last_ghost, mut state) = obs_array
            .multi_slice_mut((
                s![0, .., ..],
                s![1, .., ..],
                s![2..4, .., ..],
                s![4..8, .., ..],
                s![8..12, .., ..],
                s![12..15, .., ..],
            ));

        for (grid_value, wall_value, reward_value) in
            izip!(self.game_state.grid.iter().flatten(), wall.iter_mut(), reward.iter_mut())
        {
            *wall_value = (*grid_value == GridValue::I || *grid_value == GridValue::n) as u8 as f32;
            *reward_value = match grid_value {
                GridValue::o => variables::PELLET_SCORE,
                GridValue::O => variables::POWER_PELLET_SCORE,
                GridValue::c => variables::CHERRY_SCORE,
                _ => 0,
            } as f32
                / variables::GHOST_SCORE as f32;
        }
        for g in self.game_state.ghosts() {
            if g.is_frightened() {
                reward[g.current_pos] += 1.0;
            }
        }

        let pac_pos = self.game_state.pacbot.pos;
        pacman[(0, self.last_pos.0, self.last_pos.1)] = 1.0;
        pacman[(1, pac_pos.0, pac_pos.1)] = 1.0;

        for (i, g) in self.game_state.ghosts().enumerate() {
            let pos = g.current_pos;
            ghost[(i, pos.0, pos.1)] = 1.0;
            if g.is_frightened() {
                state[(2, pos.0, pos.1)] =
                    g.frightened_counter as f32 / variables::FRIGHTENED_LENGTH as f32;
            } else {
                let state_index =
                    if self.game_state.state == GameStateState::Chase { 1 } else { 0 };
                state[(state_index, pos.0, pos.1)] = self.game_state.state_counter as f32 / CHASE_DURATION as f32;
            }
        }

        for (i, pos) in self.last_ghost_pos.iter().enumerate() {
            last_ghost[(i, pos.0, pos.1)] = 1.0;
        }

        obs_array.slice_mut(s![15, .., ..]).fill(self.ticks_per_step as f32 / TICKS_PER_UPDATE as f32);

        obs_array
    }
}
