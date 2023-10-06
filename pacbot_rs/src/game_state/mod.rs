pub mod env;
mod py_wrappers;

use std::cell::{self, RefCell};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use pyo3::prelude::*;

use crate::{
    ghost_agent::{GhostAgent, GhostColor},
    ghost_paths::*,
    grid::{GRID, GRID_PELLET_COUNT, GRID_POWER_PELLET_COUNT},
    pacbot::PacBot,
    variables::{
        GridValue, CHERRY_POS, CHERRY_SCORE, FRIGHTENED_LENGTH, GAME_FREQUENCY, GHOST_SCORE,
        PELLET_SCORE, POWER_PELLET_SCORE, STARTING_LIVES, STATE_SWAP_TIMES, TICKS_PER_UPDATE,
    },
};

use self::py_wrappers::{wrap_ghost_agent, wrap_grid, wrap_pacbot};

const FREQUENCY: f32 = GAME_FREQUENCY * TICKS_PER_UPDATE as f32;

#[derive(Clone, Copy, Debug, Eq, PartialEq, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum GameStateState {
    Scatter = 1,
    Chase = 2,
    Frightened = 3,
}

#[derive(Clone)]
#[pyclass]
pub struct GameState {
    pub pacbot: PacBot,

    // RefCell is used here to allow GhostAgent update methods (which mutate the GhostAgent)
    // to also be able to read other GameState state.
    pub red: RefCell<GhostAgent>,
    pub pink: RefCell<GhostAgent>,
    pub orange: RefCell<GhostAgent>,
    pub blue: RefCell<GhostAgent>,

    pub grid: [[GridValue; 31]; 28],

    /// The number of remaining (regular) pellets on the grid.
    #[pyo3(get)]
    pellets: u32,
    /// The number of remaining power pellets on the grid.
    #[pyo3(get)]
    power_pellets: u32,
    /// Whether the cherry is currently on the grid.
    #[pyo3(get)]
    cherry: bool,
    prev_cherry_pellets: u32,
    old_state: GameStateState,
    pub state: GameStateState,
    pub just_swapped_state: bool,
    frightened_counter: u32,
    frightened_multiplier: u32,
    /// The current score.
    #[pyo3(get)]
    pub score: u32,
    /// Whether the game is currently playing (not paused/ended).
    #[pyo3(get)]
    pub play: bool,
    pub start_counter: u32,
    state_counter: u32,
    update_ticks: u32,
    /// The number of remaining lives.
    #[pyo3(get)]
    lives: u8,
    ticks_since_spawn: u32,
}

#[pymethods]
impl GameState {
    /// The PacBot.
    #[getter]
    fn pacbot(self_: Py<GameState>) -> impl IntoPy<Py<PyAny>> {
        wrap_pacbot(self_)
    }

    /// The red ghost.
    #[getter]
    fn red(self_: Py<GameState>) -> impl IntoPy<Py<PyAny>> {
        wrap_ghost_agent(self_, |game_state| &game_state.red)
    }

    /// The pink ghost.
    #[getter]
    fn pink(self_: Py<GameState>) -> impl IntoPy<Py<PyAny>> {
        wrap_ghost_agent(self_, |game_state| &game_state.pink)
    }

    /// The orange ghost.
    #[getter]
    fn orange(self_: Py<GameState>) -> impl IntoPy<Py<PyAny>> {
        wrap_ghost_agent(self_, |game_state| &game_state.orange)
    }

    /// The blue ghost.
    #[getter]
    fn blue(self_: Py<GameState>) -> impl IntoPy<Py<PyAny>> {
        wrap_ghost_agent(self_, |game_state| &game_state.blue)
    }

    /// The grid of tiles.
    /// Can be viewed as a NumPy array with np.asarray(...).
    #[getter]
    fn grid(self_: Py<GameState>) -> impl IntoPy<Py<PyAny>> {
        wrap_grid(self_)
    }

    /// Returns whether the current state is frightened.
    pub fn is_frightened(&self) -> bool {
        self.state == GameStateState::Frightened
    }

    /// Returns the current ghost state (scatter, chase, frightened) as an integer.
    pub fn state(&self) -> u32 {
        self.state as u32
    }

    /// Returns the frightened counter.
    pub fn frightened_counter(&self) -> u32 {
        self.frightened_counter
    }

    #[new]
    pub fn new() -> Self {
        let mut game_state = Self {
            pacbot: PacBot::new(),
            red: GhostAgent::new(
                RED_INIT_MOVES,
                GhostColor::Red,
                RED_INIT_DIR,
                &[],
                RED_SCATTER_POS,
            )
            .into(),
            pink: GhostAgent::new(
                PINK_INIT_MOVES,
                GhostColor::Pink,
                PINK_INIT_DIR,
                &PINK_START_PATH,
                PINK_SCATTER_POS,
            )
            .into(),
            orange: GhostAgent::new(
                ORANGE_INIT_MOVES,
                GhostColor::Orange,
                RED_INIT_DIR, // this is what the Python code has...?
                &ORANGE_START_PATH,
                ORANGE_SCATTER_POS,
            )
            .into(),
            blue: GhostAgent::new(
                BLUE_INIT_MOVES,
                GhostColor::Blue,
                BLUE_INIT_DIR,
                &BLUE_START_PATH,
                BLUE_SCATTER_POS,
            )
            .into(),
            grid: GRID,
            pellets: GRID_PELLET_COUNT,
            power_pellets: GRID_POWER_PELLET_COUNT,
            cherry: false,
            prev_cherry_pellets: 0,
            old_state: GameStateState::Chase,
            state: GameStateState::Scatter,
            just_swapped_state: false,
            frightened_counter: 0,
            frightened_multiplier: 1,
            score: 0,
            play: false,
            start_counter: 0,
            state_counter: 0,
            update_ticks: 0,
            lives: STARTING_LIVES,
            ticks_since_spawn: 0,
        };
        game_state.update_score();
        game_state.grid[CHERRY_POS.0][CHERRY_POS.1] = GridValue::e;
        game_state
    }

    pub fn pause(&mut self) {
        self.play = false;
    }

    pub fn unpause(&mut self) {
        self.play = true;
    }

    pub fn print_ghost_pos(&self) {
        let ghost_strings = self.ghosts().map(|g| format!("{:?}", g.current_pos));
        println!("{}", ghost_strings.collect::<Vec<_>>().join(" "));
    }

    pub fn next_step(&mut self) {
        if self.is_game_over() {
            self.end_game();
        }
        if self.should_die() {
            self.die();
        } else {
            self.check_if_ghosts_eaten();
            if self.update_ticks % TICKS_PER_UPDATE == 0 {
                self.update_ghosts();
                // This isn't in the original game code, but Pacman can actually
                // safely teleport into a ghost if you don't check twice per
                // loop
                if self.should_die() {
                    self.die();
                }
                self.check_if_ghosts_eaten();
                if self.state == GameStateState::Frightened {
                    if self.frightened_counter == 1 {
                        self.end_frightened();
                    } else if self.frightened_counter == FRIGHTENED_LENGTH {
                        self.just_swapped_state = false;
                    }
                    self.frightened_counter -= 1;
                } else {
                    self.swap_state_if_necessary();
                    self.state_counter += 1;
                }
                self.start_counter += 1;
                // self.print_ghost_pos();
            }
            self.update_score();
            if self.should_spawn_cherry() {
                self.spawn_cherry();
            }
            if self.cherry {
                self.ticks_since_spawn += 1;
            }
            if self.should_remove_cherry() {
                self.despawn_cherry();
            }
            self.update_ticks += 1;
        }
    }

    /// Sets the game back to its original state (no rounds played).
    pub fn restart(&mut self) {
        self.grid = GRID;
        self.pellets = GRID_PELLET_COUNT;
        self.power_pellets = GRID_POWER_PELLET_COUNT;
        self.cherry = false;
        self.prev_cherry_pellets = 0;
        self.old_state = GameStateState::Chase;
        self.state = GameStateState::Scatter;
        self.just_swapped_state = false;
        self.frightened_counter = 0;
        self.frightened_multiplier = 1;
        self.respawn_agents();
        self.score = 0;
        self.play = false;
        self.start_counter = 0;
        self.state_counter = 0;
        self.update_ticks = 0;
        self.lives = STARTING_LIVES;
        self.update_score();
        self.grid[CHERRY_POS.0][CHERRY_POS.1] = GridValue::e;
        self.ticks_since_spawn = 0;
    }
}

// separate impl block for private functions not exposed to Python
impl GameState {
    pub fn ghosts(&self) -> impl Iterator<Item = cell::Ref<GhostAgent>> {
        [&self.red, &self.pink, &self.orange, &self.blue]
            .into_iter()
            .map(|g| g.borrow())
    }

    pub fn ghosts_mut(&mut self) -> impl Iterator<Item = cell::RefMut<GhostAgent>> {
        [&self.red, &self.pink, &self.orange, &self.blue]
            .into_iter()
            .map(|g| g.borrow_mut())
    }

    /// Frightens all of the ghosts and saves the old state to be restored when frightened mode ends.
    fn become_frightened(&mut self) {
        if self.state != GameStateState::Frightened {
            self.old_state = self.state;
        }
        self.state = GameStateState::Frightened;
        self.frightened_counter = FRIGHTENED_LENGTH;
        self.ghosts_mut().for_each(|mut g| g.become_frightened());
        self.just_swapped_state = true;
    }

    /// Resets the state of the game to what it was before frightened,
    /// and resets the score multiplier to be equal to 1.
    fn end_frightened(&mut self) {
        self.state = self.old_state;
        self.frightened_multiplier = 1;
    }

    /// Decreases the remaining time each ghost should be frightened for and updates each ghost's
    /// current and next move information.
    fn update_ghosts(&mut self) {
        self.red.borrow_mut().update(self);
        self.orange.borrow_mut().update(self);
        self.pink.borrow_mut().update(self);
        self.blue.borrow_mut().update(self);
    }

    /// Returns true if the position of Pacman is occupied by a pellet.
    fn is_eating_pellet(&self) -> bool {
        self.grid[self.pacbot.pos.0][self.pacbot.pos.1] == GridValue::o
    }

    /// Returns true if the position of Pacman is occupied by a power pellet.
    fn is_eating_power_pellet(&self) -> bool {
        self.grid[self.pacbot.pos.0][self.pacbot.pos.1] == GridValue::O
    }

    /// Sets the current position of Pacman to empty and increments the score.
    fn eat_pellet(&mut self) {
        self.grid[self.pacbot.pos.0][self.pacbot.pos.1] = GridValue::e;
        self.score += PELLET_SCORE;
        self.pellets -= 1;
    }

    /// Sets the current position of Pacman to empty and increments the score.
    /// Also makes all ghosts frightened.
    fn eat_power_pellet(&mut self) {
        self.grid[self.pacbot.pos.0][self.pacbot.pos.1] = GridValue::e;
        self.score += POWER_PELLET_SCORE;
        self.power_pellets -= 1;
        self.become_frightened();
    }

    /// Returns true if the position of Pacman is occupied by a cherry.
    fn is_eating_cherry(&self) -> bool {
        self.grid[self.pacbot.pos.0][self.pacbot.pos.1] == GridValue::c
    }

    /// Sets the current position of Pacman to empty and increments the score.
    fn eat_cherry(&mut self) {
        self.grid[self.pacbot.pos.0][self.pacbot.pos.1] = GridValue::e;
        self.score += CHERRY_SCORE;
        self.cherry = false;
    }

    /// Returns true if the cherry should be spawned; this happens
    /// when only 170 pellets remain.
    fn should_spawn_cherry(&self) -> bool {
        (self.pellets == 170 || self.pellets == 70) && self.prev_cherry_pellets != self.pellets
    }

    fn should_remove_cherry(&self) -> bool {
        self.ticks_since_spawn >= (FREQUENCY * 10.0).ceil() as u32
    }

    /// Places the cherry on the board.
    fn spawn_cherry(&mut self) {
        self.prev_cherry_pellets = self.pellets;
        self.grid[CHERRY_POS.0][CHERRY_POS.1] = GridValue::c;
        self.cherry = true;
    }

    fn despawn_cherry(&mut self) {
        self.ticks_since_spawn = 0;
        self.grid[CHERRY_POS.0][CHERRY_POS.1] = GridValue::e;
        self.cherry = false;

        // cherry to disappear when pacman dies
    }

    /// Updates the score based on what Pacman has just eaten
    /// (what is in Pacman's current space on the board).
    fn update_score(&mut self) {
        if self.is_eating_pellet() {
            self.eat_pellet();
        }
        if self.is_eating_power_pellet() {
            self.eat_power_pellet();
        }
        if self.is_eating_cherry() {
            self.eat_cherry();
        }
    }

    /// Updates each agent's position and behavior to reflect the beginning of a new round.
    fn respawn_agents(&mut self) {
        self.pacbot.respawn();
        self.ghosts_mut().for_each(|mut g| g.respawn());
    }

    fn end_game(&mut self) {
        self.play = false;
        // println!("Score: {}", self.score);
    }

    /// Resets the round if Pacman dies with lives remaining
    /// and ends the game if Pacman has no lives remaining.
    fn die(&mut self) {
        if self.lives > 1 {
            self.respawn_agents();
            self.start_counter = 0;
            self.state_counter = 0;
            self.lives -= 1;
            self.old_state = GameStateState::Chase;
            self.state = GameStateState::Scatter;
            self.frightened_counter = 0;
            self.frightened_multiplier = 1;
            self.pause();
            self.update_score();
            self.grid[CHERRY_POS.0][CHERRY_POS.1] = GridValue::e;
        } else {
            self.end_game();
        }
    }

    /// Returns true if Pacman has collided with a ghost and the ghost is not frightened.
    fn should_die(&self) -> bool {
        self.ghosts()
            .any(|ghost| ghost.current_pos == self.pacbot.pos && ghost.frightened_counter == 0)
    }

    /// Checks for ghosts that have been eaten and sends them back
    /// to the respawn zone if they have been eaten.
    fn check_if_ghosts_eaten(&mut self) {
        for ghost in [&self.red, &self.pink, &self.orange, &self.blue] {
            // If the ghost was eaten, then the ghost is sent home, the score is updated, and
            // the score multiplier for Pacman in frightened mode is increased.
            let mut ghost = ghost.borrow_mut();
            if ghost.current_pos == self.pacbot.pos && ghost.frightened_counter > 0 {
                ghost.send_home();
                self.score += GHOST_SCORE * self.frightened_multiplier;
                self.frightened_multiplier += 1;
            }
        }
    }

    /// Returns true if there are no more pellets left on the board.
    fn are_all_pellets_eaten(&self) -> bool {
        self.pellets == 0 && self.power_pellets == 0
    }

    /// Returns true if the game is over.
    fn is_game_over(&self) -> bool {
        self.are_all_pellets_eaten()
    }

    /// Changes the state of the game
    fn swap_state_if_necessary(&mut self) {
        if STATE_SWAP_TIMES.contains(&self.state_counter) {
            if self.state == GameStateState::Chase {
                self.state = GameStateState::Scatter;
            } else {
                self.state = GameStateState::Chase;
            }
            self.just_swapped_state = true;
        } else {
            self.just_swapped_state = false;
        }
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}
