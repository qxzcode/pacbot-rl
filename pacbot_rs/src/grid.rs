use static_assertions::assert_eq_size;

use crate::variables::GridValue::{self, *};

pub const GRID: [[GridValue; 31]; 28] = include!("grid_data.txt");

pub fn is_walkable(pos: (usize, usize)) -> bool {
    let tile = GRID[pos.0][pos.1];
    tile != GridValue::I && tile != GridValue::n
}

// data computed by the build script (build.rs):

macro_rules! include_generated {
    ($filename:literal) => {
        include!(concat!(env!("OUT_DIR"), "/", $filename))
    };
}

pub const GRID_PELLET_COUNT: u32 = include_generated!("GRID_PELLET_COUNT.txt");
pub const GRID_POWER_PELLET_COUNT: u32 = include_generated!("GRID_POWER_PELLET_COUNT.txt");
pub const NUM_NODES: usize = 288;
pub const NODE_COORDS: [(usize, usize); NUM_NODES] = include_generated!("NODE_COORDS.rs");

assert_eq_size!(usize, u64);
const COORDS_TO_NODE: phf::Map<u128, usize> = include_generated!("COORDS_TO_NODE.rs");

/// Returns the node index for the given coords, or None if there is not a
/// walkable tile there.
pub fn coords_to_node(coords: (usize, usize)) -> Option<usize> {
    let key = ((coords.0 as u128) << 64) | (coords.1 as u128);
    COORDS_TO_NODE.get(&key).copied()
}

pub const EMBED_DIM: usize = 24;
#[allow(clippy::excessive_precision)]
pub const NODE_EMBEDDINGS: [[f32; EMBED_DIM]; NUM_NODES] = include_generated!("NODE_EMBEDDINGS.rs");

pub const VALID_ACTIONS: [[bool; 5]; NUM_NODES] = include_generated!("VALID_ACTIONS.rs");
pub const ACTION_DISTRIBUTIONS: [[[f32; 5]; NUM_NODES]; NUM_NODES] =
    include_generated!("ACTION_DISTRIBUTIONS.rs");
pub const DISTANCE_MATRIX: [[u8; NUM_NODES]; NUM_NODES] = include_generated!("DISTANCE_MATRIX.rs");
pub const SUPER_PELLET_LOCS: [(usize, usize); 4] = include_generated!("SUPER_PELLET_LOCS.rs");
