use crate::variables::Direction::{self, *};

// These start paths defines the paths the ghosts take at the beginning of a new round.
// Pink immedaitely leaves the respawn zone, while blue and orange remain it it for
// some time before eventually leaving. Red begins outside the zone, and as such
// does not have a predefined starting path.

pub const PINK_START_PATH: [((usize, usize), Direction); 3] =
    [((14, 17), Up), ((14, 18), Up), ((14, 19), Up)];

pub const BLUE_START_PATH: [((usize, usize), Direction); 24] = [
    ((12, 17), Up),
    ((12, 16), Down),
    ((12, 15), Down),
    ((12, 16), Up),
    ((12, 17), Up),
    ((12, 16), Down),
    ((12, 15), Down),
    ((12, 16), Up),
    ((12, 17), Up),
    ((12, 16), Down),
    ((12, 15), Down),
    ((12, 16), Up),
    ((12, 17), Up),
    ((12, 16), Down),
    ((12, 15), Down),
    ((12, 16), Up),
    ((12, 17), Up),
    ((12, 16), Down),
    ((12, 15), Down),
    ((13, 15), Right),
    ((13, 16), Up),
    ((13, 17), Up),
    ((13, 18), Up),
    ((13, 19), Up),
];

pub const ORANGE_START_PATH: [((usize, usize), Direction); 40] = [
    ((15, 17), Up),
    ((15, 16), Down),
    ((15, 15), Down),
    ((15, 16), Up),
    ((15, 17), Up),
    ((15, 16), Down),
    ((15, 15), Down),
    ((15, 16), Up),
    ((15, 17), Up),
    ((15, 16), Down),
    ((15, 15), Down),
    ((15, 16), Up),
    ((15, 17), Up),
    ((15, 16), Down),
    ((15, 15), Down),
    ((15, 16), Up),
    ((15, 17), Up),
    ((15, 16), Down),
    ((15, 15), Down),
    ((15, 16), Up),
    ((15, 17), Up),
    ((15, 16), Down),
    ((15, 15), Down),
    ((15, 16), Up),
    ((15, 17), Up),
    ((15, 16), Down),
    ((15, 15), Down),
    ((15, 16), Up),
    ((15, 17), Up),
    ((15, 16), Down),
    ((15, 15), Down),
    ((15, 16), Up),
    ((15, 17), Up),
    ((15, 16), Down),
    ((15, 15), Down),
    ((14, 15), Left),
    ((14, 16), Up),
    ((14, 17), Up),
    ((14, 18), Up),
    ((14, 19), Up),
];

/// The respawn path defines how a ghost will move out of the respawn zone when it has been
/// eaten by Pacman in the middle of a round.
pub const RESPAWN_PATH: [((usize, usize), Direction); 8] = [
    ((12, 17), Up),
    ((12, 16), Down),
    ((12, 15), Down),
    ((13, 15), Right),
    ((13, 16), Up),
    ((13, 17), Up),
    ((13, 18), Up),
    ((13, 19), Up),
];

/// This is the location where a ghost will reappear after being eaten by Pacman.
pub const GHOST_HOME_POS: (usize, usize) = (12, 15);

// These are the coordinates the ghosts attempt to move towards when they are in scatter mode.

pub const PINK_SCATTER_POS: (isize, isize) = (2, 32);
pub const ORANGE_SCATTER_POS: (isize, isize) = (0, -1);
pub const BLUE_SCATTER_POS: (isize, isize) = (27, -1);
pub const RED_SCATTER_POS: (isize, isize) = (25, 32);

// These are the locations the ghosts begin in at the start of a round.

pub const RED_INIT_MOVES: ((usize, usize), (usize, usize)) = ((13, 19), (12, 19));
pub const RED_INIT_DIR: Direction = Left;

pub const PINK_INIT_MOVES: ((usize, usize), (usize, usize)) = ((14, 15), (14, 16));
pub const PINK_INIT_DIR: Direction = Up;

pub const BLUE_INIT_MOVES: ((usize, usize), (usize, usize)) = ((12, 15), (12, 16));
pub const BLUE_INIT_DIR: Direction = Up;

pub const ORANGE_INIT_MOVES: ((usize, usize), (usize, usize)) = ((15, 15), (15, 16));
pub const ORANGE_INIT_DIR: Direction = Up;

pub const GHOST_NO_UP_TILES: [(usize, usize); 4] = [(12, 19), (15, 19), (12, 7), (15, 7)];
