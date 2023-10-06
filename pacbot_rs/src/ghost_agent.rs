use arrayvec::ArrayVec;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use rand::seq::SliceRandom;

use crate::{
    game_state::{GameState, GameStateState},
    ghost_paths::{GHOST_HOME_POS, GHOST_NO_UP_TILES, RESPAWN_PATH},
    variables::{
        Direction::{self, *},
        GridValue, FRIGHTENED_LENGTH,
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum GhostColor {
    Red = 1,
    Orange = 2,
    Pink = 3,
    Blue = 4,
}

#[derive(Clone)]
pub struct GhostAgent {
    color: GhostColor,
    init_direction: Direction,
    init_moves: ((usize, usize), (usize, usize)),
    respawn_counter: usize,
    pub start_path: &'static [((usize, usize), Direction)],
    scatter_pos: (isize, isize),
    pub frightened_counter: u32,
    pub current_pos: (usize, usize),
    pub next_pos: (usize, usize),
    direction: Direction,
}

impl GhostAgent {
    pub fn new(
        init_moves: ((usize, usize), (usize, usize)),
        color: GhostColor,
        init_direction: Direction,
        start_path: &'static [((usize, usize), Direction)],
        scatter_pos: (isize, isize),
    ) -> Self {
        Self {
            // The color of the ghost determines its movement behavior.
            color,
            init_direction,
            init_moves,
            respawn_counter: RESPAWN_PATH.len(),
            start_path,
            scatter_pos,
            frightened_counter: 0,
            current_pos: init_moves.0,
            next_pos: init_moves.1,
            direction: init_direction,
        }
    }

    fn is_move_legal(&self, move_pos: (usize, usize), game_state: &GameState) -> bool {
        move_pos != self.current_pos
            && game_state.grid[move_pos.0][move_pos.1] != GridValue::I
            && game_state.grid[move_pos.0][move_pos.1] != GridValue::n
    }

    /// Returns a list of valid tiles for the ghost to move to. If no such tiles exist,
    /// return a list containing only the ghost's current position.
    fn possible_moves(&self, game_state: &GameState) -> ArrayVec<(usize, usize), 4> {
        let (x, y) = self.next_pos;
        let mut possible = ArrayVec::<(usize, usize), 4>::new();
        if self.is_move_legal((x + 1, y), game_state) {
            possible.push((x + 1, y));
        }
        if self.is_move_legal((x, y + 1), game_state) && !GHOST_NO_UP_TILES.contains(&(x, y)) {
            possible.push((x, y + 1));
        }
        if self.is_move_legal((x - 1, y), game_state) {
            possible.push((x - 1, y));
        }
        if self.is_move_legal((x, y - 1), game_state) {
            possible.push((x, y - 1));
        }
        if possible.is_empty() {
            possible.push(self.current_pos);
        }
        possible
    }

    /// Returns the direction of the ghost based on its previous coordinates.
    fn get_direction(&self, pos_prev: (usize, usize), pos_new: (usize, usize)) -> Direction {
        if pos_new.0 > pos_prev.0 {
            Right
        } else if pos_new.0 < pos_prev.0 {
            Left
        } else if pos_new.1 > pos_prev.1 {
            Up
        } else if pos_new.1 < pos_prev.1 {
            Down
        } else {
            self.direction
        }
    }

    /// This is awful. Look online to find out blue is supposed to move, and let's just work under the
    /// assumption that this function returns that sort of move.
    fn get_next_blue_chase_move(&self, game_state: &GameState) -> ((usize, usize), Direction) {
        let pacbot_x = game_state.pacbot.pos.0 as isize;
        let pacbot_y = game_state.pacbot.pos.1 as isize;
        let pacbot_target = match game_state.pacbot.direction {
            Up => (pacbot_x - 2, pacbot_y + 2),
            Down => (pacbot_x, pacbot_y - 2),
            Left => (pacbot_x - 2, pacbot_y),
            Right => (pacbot_x + 2, pacbot_y),
        };
        let red_pos = game_state.red.borrow().current_pos;
        let x = pacbot_target.0 + (pacbot_target.0 - red_pos.0 as isize);
        let y = pacbot_target.1 + (pacbot_target.1 - red_pos.1 as isize);

        self.get_move_based_on_target((x, y), game_state)
    }

    /// Return the move closest to the space 4 tiles ahead of Pacman in the direction
    /// Pacman is currently facing. If Pacman is facing up, then we replicate a bug in
    /// the original game and return the move closest to the space 4 tiles above and
    /// 4 tiles to the left of Pacman.
    fn get_next_pink_chase_move(&self, game_state: &GameState) -> ((usize, usize), Direction) {
        let pacbot_x = game_state.pacbot.pos.0 as isize;
        let pacbot_y = game_state.pacbot.pos.1 as isize;
        let target_pos = match game_state.pacbot.direction {
            Up => (pacbot_x - 4, pacbot_y + 4),
            Down => (pacbot_x, pacbot_y - 4),
            Left => (pacbot_x - 4, pacbot_y),
            Right => (pacbot_x + 4, pacbot_y),
        };

        self.get_move_based_on_target(target_pos, game_state)
    }

    /// Returns the move that will bring the ghost closest to Pacman
    fn get_next_red_chase_move(&self, game_state: &GameState) -> ((usize, usize), Direction) {
        self.get_move_based_on_target(game_state.pacbot.pos, game_state)
    }

    /// If the ghost is close to Pacman, return the move that will bring the ghost closest
    /// to its scatter position (bottom left corner). If the ghost is far from Pacman,
    /// return the move that will bring the ghost closest to Pacman.
    fn get_next_orange_chase_move(&self, game_state: &GameState) -> ((usize, usize), Direction) {
        if squared_euclidian_distance(self.current_pos, game_state.pacbot.pos) < 8 * 8 {
            self.get_next_scatter_move(game_state)
        } else {
            self.get_move_based_on_target(game_state.pacbot.pos, game_state)
        }
    }

    /// Moves to the tile that is the closest to the target by straight-line distance,
    /// NOT the tile that is the closest to the target by optimal tile path length.
    fn get_move_based_on_target<T: AsISize>(
        &self,
        target_pos: (T, T),
        game_state: &GameState,
    ) -> ((usize, usize), Direction) {
        let min_move = self
            .possible_moves(game_state)
            .into_iter()
            .min_by_key(|pos| squared_euclidian_distance(target_pos, *pos))
            .unwrap();
        (min_move, self.get_direction(self.next_pos, min_move))
    }

    /// Returns the correct chase mode move for the ghost.
    fn get_next_chase_move(&self, game_state: &GameState) -> ((usize, usize), Direction) {
        match self.color {
            GhostColor::Blue => self.get_next_blue_chase_move(game_state),
            GhostColor::Pink => self.get_next_pink_chase_move(game_state),
            GhostColor::Red => self.get_next_red_chase_move(game_state),
            GhostColor::Orange => self.get_next_orange_chase_move(game_state),
        }
    }

    /// Returns the move that will bring the ghost closest to its scatter position.
    /// Since the state (chase/scatter) is the same for all ghosts, it is stored as part of the game state.
    fn get_next_scatter_move(&self, game_state: &GameState) -> ((usize, usize), Direction) {
        self.get_move_based_on_target(self.scatter_pos, game_state)
    }

    /// Returns a random move selected from the list of valid moves for this ghost.
    fn get_next_frightened_move(&self, game_state: &GameState) -> ((usize, usize), Direction) {
        let move_pos = *self
            .possible_moves(game_state)
            .choose(&mut rand::thread_rng())
            .unwrap();
        (move_pos, self.get_direction(self.next_pos, move_pos))
    }

    fn reverse_direction(&self) -> ((usize, usize), Direction) {
        let move_pos = self.current_pos;
        (move_pos, self.get_direction(self.next_pos, move_pos))
    }

    /// Returns the correct move for the ghost based on what state the ghost is in.
    fn get_next_state_move(&self, game_state: &GameState) -> ((usize, usize), Direction) {
        if game_state.just_swapped_state {
            self.reverse_direction()
        } else if self.frightened_counter > 0 {
            self.get_next_frightened_move(game_state)
        } else if game_state.state == GameStateState::Chase {
            self.get_next_chase_move(game_state)
        } else {
            self.get_next_scatter_move(game_state)
        }
    }

    /// Returns true if a round has just started; in this case, the ghost should follow
    /// its predefined starting path.
    fn should_follow_starting_path(&self, game_state: &GameState) -> bool {
        (game_state.start_counter as usize) < self.start_path.len()
    }

    /// Returns true if the ghost has just respawned and should follow the predefined
    /// respawn path such that the ghost leaves the respawn zone.
    fn should_follow_respawn_path(&self) -> bool {
        self.respawn_counter < RESPAWN_PATH.len()
    }

    /// Returns the correct move for the ghost based on the state of the game: namely, it
    /// will return a move based on if a round has just begun, if the ghost has just respawned,
    /// or if the ghost should be acting normally.
    fn decide_next_moves(&mut self, game_state: &GameState) -> ((usize, usize), Direction) {
        if self.should_follow_starting_path(game_state) {
            self.start_path[game_state.start_counter as usize]
        } else if self.should_follow_respawn_path() {
            self.respawn_counter += 1;
            RESPAWN_PATH[self.respawn_counter - 1]
        } else {
            self.get_next_state_move(game_state)
        }
    }

    /// Decreases the remaining time the ghost should be frightened for and updates the ghost's
    /// current and next move information.
    pub fn update(&mut self, game_state: &GameState) {
        if self.frightened_counter > 0 {
            self.frightened_counter -= 1;
        }
        let (next_pos, next_dir) = self.decide_next_moves(game_state);
        self.current_pos = self.next_pos;
        self.next_pos = next_pos;
        self.direction = next_dir;
    }

    /// Sets the ghost's position back to the respawn zone and removes the frightened condition.
    /// This function is called when the ghost gets eaten by Pacman.
    pub fn send_home(&mut self) {
        self.current_pos = GHOST_HOME_POS;
        self.next_pos = (GHOST_HOME_POS.0, GHOST_HOME_POS.1 + 1);
        self.direction = Up;
        // This will make the ghost follow its respawn path, ensuring it leaves the respawn zone.
        self.respawn_counter = 0;
        self.frightened_counter = 0;
    }

    /// Sets the remaining amount of frames the ghost will be frightened for to
    /// the number of game updates the ghost should stay frightened for when a power pellet
    /// is eaten. This makes the ghost frightened if it was not already.
    pub fn become_frightened(&mut self) {
        self.frightened_counter = FRIGHTENED_LENGTH;
    }

    /// Returns true if the ghost is frightened.
    pub fn is_frightened(&self) -> bool {
        self.frightened_counter > 0
    }

    pub fn respawn(&mut self) {
        self.current_pos = self.init_moves.0;
        self.next_pos = self.init_moves.1;
        self.direction = self.init_direction;
        self.frightened_counter = 0;
        // This will prevent the ghost from following the respawn path it follows when
        // leaving the start area AFTER being eaten by Pacman during a round.
        self.respawn_counter = RESPAWN_PATH.len();
    }
}

trait AsISize: Copy {
    fn as_isize(self) -> isize;
}
impl AsISize for isize {
    fn as_isize(self) -> isize {
        self
    }
}
impl AsISize for usize {
    fn as_isize(self) -> isize {
        self as isize
    }
}
fn squared_euclidian_distance<T1: AsISize, T2: AsISize>(pos1: (T1, T1), pos2: (T2, T2)) -> usize {
    let dx = pos1.0.as_isize() - pos2.0.as_isize();
    let dy = pos1.1.as_isize() - pos2.1.as_isize();
    (dx * dx + dy * dy) as usize
}
