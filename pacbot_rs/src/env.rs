use crate::grid::{coords_to_node, NODE_COORDS, VALID_ACTIONS};
use ndarray::{s, Array, Array3};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use numpy::{IntoPyArray, PyArray3};
use pacbot_rs_2::game_modes::GameMode;
use pacbot_rs_2::game_state::GameState;
use pacbot_rs_2::location::{Direction::*, LocationState};
use pacbot_rs_2::variables::{self, GHOST_FRIGHT_STEPS, INIT_LEVEL};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{seq::SliceRandom, Rng};

#[derive(Copy, Clone, Debug, Default)]
pub struct PacmanGymConfiguration {
    pub random_start: bool,
    pub random_ticks: bool,
    /// Whether to randomize the ghosts' positions when `random_start = true`.
    pub randomize_ghosts: bool,
    /// Whether to remove super pellets from the board before starting the game
    pub remove_super_pellets: bool,
}

pub const OBS_SHAPE: (usize, usize, usize) = (17, 28, 31);

pub const TICKS_PER_UPDATE: u32 = 12;
/// How many ticks the game should move every step normally. Ghosts move every 12 ticks.
const NORMAL_TICKS_PER_STEP: u32 = 8;

/// Minimum number of ticks per step.
const MIN_TICKS_PER_STEP: u32 = 4;

/// Maximum number of ticks per step.
const MAX_TICKS_PER_STEP: u32 = 14;

/// Penalty for turning.
const TURN_PENALTY: i32 = -2;

const LAST_REWARD: u16 = 3_000;

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
    pub game_state: GameState,
    #[pyo3(get, set)]
    pub random_start: bool,
    last_score: u16,
    last_action: Action,
    /// Position of ghosts on the last frame, in obs coords.
    last_ghost_pos: [Option<(usize, usize)>; 4],
    /// Position of Pacman on the last frame, in obs coords.
    last_pos: Option<(usize, usize)>,
    ticks_per_step: u32,
    random_ticks: bool,
    randomize_ghosts: bool,
    remove_super_pellets: bool,
}

fn modify_bit_u32(num: &mut u32, bit_idx: usize, bit_val: bool) {
    // If the bit is true, we should set the bit, otherwise we clear it
    if bit_val {
        *num |= 1 << bit_idx;
    } else {
        *num &= !(1 << bit_idx);
    }
}

/// Converts game location into our coords.
fn loc_to_pos(loc: LocationState) -> Option<(usize, usize)> {
    if loc.row != 32 && loc.col != 32 {
        Some((loc.col as usize, (31 - loc.row - 1) as usize))
    } else {
        None
    }
}

#[pymethods]
impl PacmanGym {
    #[new]
    pub fn new(random_start: bool, random_ticks: bool) -> Self {
        let game_state = GameState { paused: false, ..GameState::new() };
        let configuration =
            PacmanGymConfiguration { random_start, random_ticks, ..Default::default() };
        Self::new_with_state(configuration, game_state)
    }

    pub fn reset(&mut self) {
        self.last_score = 0;
        self.game_state = GameState::new();

        self.apply_configuration();

        let rng = &mut rand::thread_rng();
        if self.random_ticks {
            self.ticks_per_step = rng.gen_range(MIN_TICKS_PER_STEP..MAX_TICKS_PER_STEP);
        }

        if self.random_start {
            let mut random_pos = || *NODE_COORDS.choose(rng).unwrap();

            let pac_random_pos = random_pos();
            self.game_state.pacman_loc = LocationState::new(pac_random_pos.0, pac_random_pos.1, Up);

            if self.randomize_ghosts {
                for ghost in &mut self.game_state.ghosts {
                    ghost.trapped_steps = 0;
                    let ghost_random_pos = random_pos();
                    // find a valid next space
                    let index = coords_to_node(ghost_random_pos).expect("invalid random pos!");
                    let valid_moves = VALID_ACTIONS[index]
                        .iter()
                        .enumerate()
                        .filter(|(_, b)| **b)
                        .nth(1)
                        .unwrap();
                    ghost.loc = LocationState {
                        row: ghost_random_pos.0,
                        col: ghost_random_pos.1,
                        dir: match valid_moves.0 {
                            1 => Down,
                            2 => Up,
                            3 => Right,
                            4 => Left,
                            _ => unreachable!(),
                        },
                    };
                    ghost.next_loc = ghost.loc;
                }
            }

            // Randomly remove pellets from half the board (left, right, top, bottom) or don't.
            let wipe_type = rng.gen_range(0..=4);
            if wipe_type != 0 {
                for row in 0..31 {
                    for col in 0..28 {
                        if self.game_state.pellet_at((row, col))
                            && match wipe_type {
                                1 => col < 28 / 2,
                                2 => col >= 28 / 2,
                                3 => row < 31 / 2,
                                4 => row >= 31 / 2,
                                _ => unreachable!(),
                            }
                        {
                            modify_bit_u32(
                                &mut self.game_state.pellets[row as usize],
                                col as usize,
                                false,
                            );
                            self.game_state.decrement_num_pellets();
                        }
                    }
                }
            }
        }

        self.last_ghost_pos = [
            loc_to_pos(self.game_state.ghosts[0].loc),
            loc_to_pos(self.game_state.ghosts[1].loc),
            loc_to_pos(self.game_state.ghosts[2].loc),
            loc_to_pos(self.game_state.ghosts[3].loc),
        ];
        self.last_action = Action::Stay;
        self.last_pos = None;

        self.game_state.paused = false;
    }

    /// Performs an action and steps the environment.
    /// Returns (reward, done).
    pub fn step(&mut self, action: Action) -> (i32, bool) {
        // Update Pacman pos
        self.last_pos = loc_to_pos(self.game_state.pacman_loc);
        self.move_one_cell(action);

        let entity_positions = [
            loc_to_pos(self.game_state.ghosts[0].loc),
            loc_to_pos(self.game_state.ghosts[1].loc),
            loc_to_pos(self.game_state.ghosts[2].loc),
            loc_to_pos(self.game_state.ghosts[3].loc),
        ];

        // step through environment multiple times
        let turn_penalty = if self.last_action == action || self.last_action == Action::Stay {
            0
        } else {
            TURN_PENALTY
        };
        for _ in 0..self.ticks_per_step {
            self.game_state.step();
            if self.is_done() {
                break;
            }
        }
        self.last_action = action;

        let game_state = &self.game_state;

        // If the ghost positions change, update the last ghost positions
        let new_entity_positions = [
            loc_to_pos(game_state.ghosts[0].loc),
            loc_to_pos(game_state.ghosts[1].loc),
            loc_to_pos(game_state.ghosts[2].loc),
            loc_to_pos(game_state.ghosts[3].loc),
        ];
        let pos_changed =
            entity_positions.iter().zip(&new_entity_positions).any(|(e1, e2)| e1 != e2);
        if pos_changed {
            self.last_ghost_pos = entity_positions;
        }

        let done = self.is_done();

        // The reward is raw difference in game score, minus a penalty for dying or
        // plus a bonus for clearing the board.
        let mut reward = (game_state.curr_score as i32 - self.last_score as i32) + turn_penalty;
        if done {
            if game_state.curr_lives < 3 {
                // Pacman died.
                reward += -200;
            } else {
                // Pacman cleared the board! Good Pacman.
                reward += LAST_REWARD as i32;
            }
        }
        // Punishment for being too close to a ghost
        let pacbot_pos = self.game_state.pacman_loc;
        for ghost in &game_state.ghosts {
            if !ghost.is_frightened() {
                let ghost_pos = ghost.loc;
                reward += match (pacbot_pos.row - ghost_pos.row).abs()
                    + (pacbot_pos.col - ghost_pos.col).abs()
                {
                    1 => -50,
                    2 => -20,
                    _ => 0,
                };
            }
        }
        self.last_score = game_state.curr_score;

        (reward, done)
    }

    pub fn score(&self) -> u32 {
        self.game_state.curr_score as u32
    }

    pub fn lives(&self) -> u8 {
        self.game_state.get_lives()
    }

    pub fn is_done(&self) -> bool {
        self.game_state.get_lives() < 3 || self.game_state.get_level() != INIT_LEVEL
    }

    pub fn first_ai_done(&self) -> bool {
        // are super pellets gone and ghosts not frightened? then switch models
        [(3, 1), (3, 26), (23, 1), (23, 26)].into_iter().all(|x| !self.game_state.pellet_at(x))
            && self.game_state.ghosts.iter().all(|g| !g.is_frightened())
    }

    pub fn remaining_pellets(&self) -> u16 {
        self.game_state.get_num_pellets()
    }

    /// Returns the action mask that is `True` for currently-valid actions and
    /// `False` for currently-invalid actions.
    pub fn action_mask(&self) -> [bool; 5] {
        let p = self.game_state.pacman_loc;
        [
            true,
            !self.game_state.wall_at((p.row + 1, p.col)),
            !self.game_state.wall_at((p.row - 1, p.col)),
            !self.game_state.wall_at((p.row, p.col - 1)),
            !self.game_state.wall_at((p.row, p.col + 1)),
        ]
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

        let game_state = &self.game_state;

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
                    } else if (row, col) == (game_state.fruit_loc.row, game_state.fruit_loc.col)
                        && game_state.fruit_exists()
                    {
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
    pub fn new_with_state(configuration: PacmanGymConfiguration, game_state: GameState) -> Self {
        let last_ghost_pos = [
            loc_to_pos(game_state.ghosts[0].loc),
            loc_to_pos(game_state.ghosts[1].loc),
            loc_to_pos(game_state.ghosts[2].loc),
            loc_to_pos(game_state.ghosts[3].loc),
        ];
        let mut s = Self {
            random_start: configuration.random_start,
            last_score: 0,
            last_action: Action::Stay,
            last_ghost_pos,
            last_pos: loc_to_pos(game_state.pacman_loc),
            game_state,
            ticks_per_step: NORMAL_TICKS_PER_STEP,
            random_ticks: configuration.random_ticks,
            randomize_ghosts: configuration.randomize_ghosts,
            remove_super_pellets: configuration.remove_super_pellets,
        };
        s.apply_configuration();
        s
    }

    fn apply_configuration(&mut self) {
        if self.remove_super_pellets {
            for (row, col) in [(3, 1), (23, 1), (3, 26), (23, 26)] {
                modify_bit_u32(&mut self.game_state.pellets[row as usize], col as usize, false);
                self.game_state.decrement_num_pellets();
            }
        }
    }

    pub fn set_state(&mut self, new_state: GameState) {
        // Update Pacman pos
        self.last_pos = loc_to_pos(self.game_state.pacman_loc);
        self.game_state = new_state;

        self.apply_configuration();

        let entity_positions = [
            loc_to_pos(self.game_state.ghosts[0].loc),
            loc_to_pos(self.game_state.ghosts[1].loc),
            loc_to_pos(self.game_state.ghosts[2].loc),
            loc_to_pos(self.game_state.ghosts[3].loc),
        ];

        self.last_action = Action::Stay;

        let game_state = &self.game_state;

        // If the ghost positions change, update the last ghost positions
        let new_entity_positions = [
            loc_to_pos(game_state.ghosts[0].loc),
            loc_to_pos(game_state.ghosts[1].loc),
            loc_to_pos(game_state.ghosts[2].loc),
            loc_to_pos(game_state.ghosts[3].loc),
        ];
        let pos_changed =
            entity_positions.iter().zip(&new_entity_positions).any(|(e1, e2)| e1 != e2);
        if pos_changed {
            self.last_ghost_pos = entity_positions;
        }

        self.last_score = game_state.curr_score;
    }

    fn move_one_cell(&mut self, action: Action) {
        let old_pos = self.game_state.pacman_loc;
        let new_pos = match action {
            Action::Stay => (old_pos.row, old_pos.col),
            Action::Right => (old_pos.row, old_pos.col + 1),
            Action::Left => (old_pos.row, old_pos.col.saturating_sub(1)),
            Action::Up => (old_pos.row.saturating_sub(1), old_pos.col),
            Action::Down => (old_pos.row + 1, old_pos.col),
        };
        if !self.game_state.wall_at(new_pos) {
            self.game_state.set_pacman_location(new_pos);
        }
    }

    /// Returns an observation array/tensor constructed from the game state.
    pub fn obs(&self) -> Array3<f32> {
        let game_state = &self.game_state;
        let mut obs_array = Array::zeros(OBS_SHAPE);
        let (mut wall, mut reward, mut pacman, mut ghost, mut last_ghost, mut state) = obs_array
            .multi_slice_mut((
                s![0, .., ..],
                s![1, .., ..],
                s![2..4, .., ..],
                s![4..8, .., ..],
                s![8..12, .., ..],
                s![12..15, .., ..],
            ));

        for row in 0..31 {
            for col in 0..28 {
                let obs_row = 31 - row - 1;
                wall[(col, obs_row)] = game_state.wall_at((row as i8, col as i8)) as u8 as f32;
                reward[(col, obs_row)] = if game_state.pellet_at((row as i8, col as i8)) {
                    (if ((row == 3) || (row == 23)) && ((col == 1) || (col == 26)) {
                        variables::SUPER_PELLET_POINTS
                    } else {
                        variables::PELLET_POINTS
                    }) + (if game_state.num_pellets == 1 { LAST_REWARD } else { 0 })
                } else if game_state.fruit_exists()
                    && col == game_state.fruit_loc.col as usize
                    && row == game_state.fruit_loc.row as usize
                {
                    variables::FRUIT_POINTS
                } else {
                    0
                } as f32
                    / variables::COMBO_MULTIPLIER as f32;
            }
        }

        // Compute new pacman and ghost positions
        let new_pos = loc_to_pos(game_state.pacman_loc);
        let new_ghost_pos: Vec<_> = game_state.ghosts.iter().map(|g| loc_to_pos(g.loc)).collect();

        if let Some(last_pos) = self.last_pos {
            pacman[(0, last_pos.0, last_pos.1)] = 1.0;
        }
        if let Some(new_pos) = new_pos {
            pacman[(1, new_pos.0, new_pos.1)] = 1.0;
        }

        for (i, g) in game_state.ghosts.iter().enumerate() {
            if let Some((col, row)) = new_ghost_pos[i] {
                ghost[(i, col, row)] = 1.0;
                if g.is_frightened() {
                    state[(2, col, row)] = g.fright_steps as f32 / GHOST_FRIGHT_STEPS as f32;
                    reward[(col, row)] += 2_i32.pow(game_state.ghost_combo as u32) as f32;
                } else {
                    let state_index = if game_state.mode == GameMode::CHASE { 1 } else { 0 };
                    state[(state_index, col, row)] =
                        game_state.get_mode_steps() as f32 / GameMode::CHASE.duration() as f32;
                }
            }
        }

        for (i, pos) in self.last_ghost_pos.iter().enumerate() {
            if let Some(pos) = pos {
                last_ghost[(i, pos.0, pos.1)] = 1.0;
            }
        }

        obs_array
            .slice_mut(s![15, .., ..])
            .fill(self.ticks_per_step as f32 / game_state.get_update_period() as f32);

        // Super pellet map
        for row in 0..31 {
            for col in 0..28 {
                let obs_row = 31 - row - 1;
                if game_state.pellet_at((row as i8, col as i8))
                    && ((row == 3) || (row == 23))
                    && ((col == 1) || (col == 26))
                {
                    obs_array[(16, col, obs_row)] = 1.;
                }
            }
        }

        obs_array
    }
}
