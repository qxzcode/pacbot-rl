use crate::grid::{coords_to_node, NODE_COORDS, VALID_ACTIONS};
use itertools::izip;
use ndarray::{s, Array, Array3};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use numpy::{IntoPyArray, PyArray3};
use pacbot_rs_2::game_engine::GameEngine;
use pacbot_rs_2::game_modes::GameMode;
use pacbot_rs_2::location::{LocationState, DOWN, LEFT, RIGHT, UP};
use pacbot_rs_2::variables::{
    COMBO_MULTIPLIER, FRUIT_POINTS, GHOST_FRIGHT_STEPS, PELLET_POINTS, SUPER_PELLET_POINTS,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{seq::SliceRandom, Rng};

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
    pub game_engine: GameEngine,
    #[pyo3(get, set)]
    pub random_start: bool,
    last_score: u16,
    last_action: Action,
    last_ghost_pos: [LocationState; 4],
    last_pos: LocationState,
}

fn modify_bit_u32(num: &mut u32, bit_idx: usize, bit_val: bool) {
    // If the bit is true, we should set the bit, otherwise we clear it
    if bit_val {
        *num |= 1 << bit_idx;
    } else {
        *num &= !(1 << bit_idx);
    }
}

#[pymethods]
impl PacmanGym {
    #[new]
    pub fn new(random_start: bool) -> Self {
        let mut game_engine = GameEngine::new();
        game_engine.unpause();
        let last_ghost_pos = [
            game_engine.get_state().ghosts[0].loc,
            game_engine.get_state().ghosts[1].loc,
            game_engine.get_state().ghosts[2].loc,
            game_engine.get_state().ghosts[3].loc,
        ];
        let mut env = Self {
            random_start,
            last_score: 0,
            last_action: Action::Stay,
            last_ghost_pos,
            last_pos: game_engine.get_state().pacman_loc,
            game_engine,
        };
        if random_start && RANDOMIZE_GHOSTS {
            for ghost in &mut env.game_engine.state_mut().ghosts {
                ghost.trapped_steps = 0;
            }
        }
        env
    }

    pub fn reset(&mut self) {
        self.last_score = 0;
        self.game_engine = GameEngine::new();

        let game_state = self.game_engine.state_mut();

        if self.random_start {
            let rng = &mut rand::thread_rng();
            let mut random_pos = || *NODE_COORDS.choose(rng).unwrap();

            let pac_random_pos = random_pos();
            game_state.pacman_loc = LocationState::new(pac_random_pos.0, pac_random_pos.1, 0);

            if RANDOMIZE_GHOSTS {
                for ghost in &mut game_state.ghosts {
                    ghost.trapped_steps = 0;
                    let ghost_random_pos = random_pos();
                    ghost.loc =
                        LocationState { row: ghost_random_pos.0, col: ghost_random_pos.1, dir: 0 };
                    ghost.next_loc = ghost.loc;
                }
            }

            // Randomly remove pellets from half the board (left, right, top, bottom) or don't.
            let wipe_type = rng.gen_range(0..=4);
            if wipe_type != 0 {
                for row in 0..31 {
                    for col in 0..28 {
                        if game_state.pellet_at((row, col))
                            && match wipe_type {
                                1 => col < 28 / 2,
                                2 => col >= 28 / 2,
                                3 => row < 31 / 2,
                                4 => row >= 31 / 2,
                                _ => unreachable!(),
                            }
                        {
                            modify_bit_u32(
                                &mut game_state.pellets[row as usize],
                                col as usize,
                                false,
                            );
                            game_state.decrement_num_pellets();
                        }
                    }
                }
            }
        }

        self.last_ghost_pos = [
            game_state.ghosts[0].loc,
            game_state.ghosts[1].loc,
            game_state.ghosts[2].loc,
            game_state.ghosts[3].loc,
        ];
        self.last_action = Action::Stay;
        self.last_pos = game_state.pacman_loc;

        self.game_engine.unpause();
    }

    /// Performs an action and steps the environment.
    /// Returns (reward, done).
    pub fn step(&mut self, action: Action) -> (i32, bool) {
        // update Pacman pos
        self.last_pos = self.game_engine.get_state().pacman_loc;
        self.move_one_cell(action);

        let game_state = self.game_engine.state_mut();

        let entity_positions = [
            game_state.ghosts[0].loc,
            game_state.ghosts[1].loc,
            game_state.ghosts[2].loc,
            game_state.ghosts[3].loc,
        ];

        // step through environment multiple times
        // If changing directions, double the number of ticks
        let tick_mult =
            if self.last_action == action || self.last_action == Action::Stay { 1 } else { 2 };
        for _ in 0..TICKS_PER_STEP * tick_mult {
            self.game_engine.step();
            if self.is_done() {
                break;
            }
        }
        self.last_action = action;

        let game_state = self.game_engine.get_state();

        // If the ghost positions change, update the last ghost positions
        let new_entity_positions = [
            game_state.ghosts[0].loc,
            game_state.ghosts[1].loc,
            game_state.ghosts[2].loc,
            game_state.ghosts[3].loc,
        ];
        let pos_changed =
            entity_positions.iter().zip(&new_entity_positions).any(|(e1, e2)| e1 != e2);
        if pos_changed {
            self.last_ghost_pos = entity_positions;
        }

        let done = self.is_done();

        // The reward is raw difference in game score, minus a penalty for dying or
        // plus a bonus for clearing the board.
        let mut reward = game_state.curr_score as i32 - self.last_score as i32;
        if done {
            if game_state.curr_lives < 3 {
                // Pacman died.
                reward += -200;
            } else {
                // Pacman cleared the board! Good Pacman.
                reward += 3_000;
            }
        }
        self.last_score = game_state.curr_score;

        (reward, done)
    }

    pub fn score(&self) -> u32 {
        self.game_engine.get_state().curr_score as u32
    }

    pub fn lives(&self) -> u8 {
        self.game_engine.get_state().get_lives()
    }

    pub fn is_done(&self) -> bool {
        self.game_engine.get_state().get_lives() < 3
    }

    pub fn remaining_pellets(&self) -> u16 {
        self.game_engine.get_state().get_num_pellets()
    }

    /// Returns the action mask that is `True` for currently-valid actions and
    /// `False` for currently-invalid actions.
    pub fn action_mask(&self) -> [bool; 5] {
        let pacbot_pos = self.game_engine.get_state().pacman_loc;
        let pacbot_node = coords_to_node((pacbot_pos.row, pacbot_pos.col))
            .expect("PacBot is in an invalid location");
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

        let game_state = self.game_engine.get_state();

        // Print the game grid.
        let ghost_char = |x, y| {
            for (i, ch) in ['R', 'P', 'B', 'O'].iter().enumerate() {
                if (x, y) == (game_state.ghosts[i].loc.row, game_state.ghosts[i].loc.col) {
                    let color =
                        if game_state.ghosts[i].is_frightened() { "96" } else { "38;5;206" };
                    return Some((*ch, color));
                }
            }
            None
        };
        for row in 0..31 {
            for col in 0..28 {
                let (ch, style) =
                    if (row, col) == (game_state.pacman_loc.row, game_state.pacman_loc.col) {
                        ('@', "93")
                    } else if let Some(ch) = ghost_char(row, col) {
                        ch
                    } else if game_state.wall_at((row, col)) {
                        ('#', "90")
                    } else if (row, col) == (game_state.fruit_loc.row, game_state.fruit_loc.col) {
                        ('c', "31")
                    } else if game_state.pellet_at((row, col)) {
                        if ((row == 3) || (row == 23)) && ((col == 1) || (col == 26)) {
                            // super pellet
                            ('o', "")
                        } else {
                            ('.', "")
                        }
                    } else {
                        (' ', "")
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
        let old_pos = self.game_engine.get_state().pacman_loc;
        let new_pos = match action {
            Action::Stay => (old_pos.row, old_pos.col),
            Action::Right => (old_pos.row, min(old_pos.col + 1, 31 - 1)),
            Action::Up => (old_pos.row, old_pos.col.saturating_sub(1)),
            Action::Left => (old_pos.row.saturating_sub(1), old_pos.col),
            Action::Down => (min(old_pos.row + 1, 28 - 1), old_pos.col),
        };
        if !self.game_engine.get_state().wall_at(new_pos) {
            let old_pos = self.game_engine.get_state().pacman_loc;
            self.game_engine.set_pacman_location(LocationState {
                row: new_pos.0,
                col: new_pos.1,
                dir: if old_pos.row < new_pos.0 {
                    DOWN
                } else if old_pos.row > new_pos.0 {
                    UP
                } else if old_pos.col < new_pos.1 {
                    RIGHT
                } else {
                    LEFT
                },
            });
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

        let game_state = self.game_engine.get_state();
        for (grid_value, wall_value, reward_value) in izip!(
            (0..31).map(|row| (0..28).map(move |col| (row, col))).flatten(),
            wall.iter_mut(),
            reward.iter_mut()
        ) {
            *wall_value = game_state.wall_at(grid_value) as u8 as f32;

            *reward_value = if grid_value == (game_state.fruit_loc.row, game_state.fruit_loc.col) {
                FRUIT_POINTS
            } else if game_state.pellet_at(grid_value) {
                if ((grid_value.0 == 3) || (grid_value.0 == 23))
                    && ((grid_value.1 == 1) || (grid_value.1 == 26))
                {
                    SUPER_PELLET_POINTS
                } else {
                    PELLET_POINTS
                }
            } else {
                0
            } as f32
                / COMBO_MULTIPLIER as f32;
        }
        for g in &self.game_engine.get_state().ghosts {
            if g.is_frightened() {
                reward[(g.loc.col as usize, g.loc.row as usize)] += 1.0;
            }
        }

        let pac_pos = self.game_engine.get_state().pacman_loc;
        pacman[(0, self.last_pos.col as usize, self.last_pos.row as usize)] = 1.0;
        pacman[(1, pac_pos.col as usize, pac_pos.row as usize)] = 1.0;

        for (i, g) in self.game_engine.get_state().ghosts.iter().enumerate() {
            let pos = g.loc;
            if pos.row == 32 && pos.col == 32 {
                continue;
            }
            ghost[(i, pos.col as usize, pos.row as usize)] = 1.0;
            if g.is_frightened() {
                state[(2, pos.col as usize, pos.row as usize)] =
                    g.fright_steps as f32 / GHOST_FRIGHT_STEPS as f32;
            } else {
                let state_index =
                    if self.game_engine.get_state().mode == GameMode::CHASE { 1 } else { 0 };
                state[(state_index, pos.col as usize, pos.row as usize)] = 1.0;
            }
        }

        for (i, pos) in self.last_ghost_pos.iter().enumerate() {
            if pos.row == 32 && pos.col == 32 {
                continue;
            }
            last_ghost[(i, pos.col as usize, pos.row as usize)] = 1.0;
        }

        obs_array
    }
}
