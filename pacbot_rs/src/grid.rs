use static_assertions::assert_eq_size;

// data computed by the build script (build.rs):

macro_rules! include_generated {
    ($filename:literal) => {
        include!(concat!(env!("OUT_DIR"), "/", $filename))
    };
}

pub const GRID_PELLET_COUNT: u32 = include_generated!("GRID_PELLET_COUNT.txt");
pub const GRID_POWER_PELLET_COUNT: u32 = include_generated!("GRID_POWER_PELLET_COUNT.txt");
pub const NUM_NODES: usize = 288;
pub const NODE_COORDS: [(i8, i8); NUM_NODES] = include_generated!("NODE_COORDS.rs");

assert_eq_size!(usize, u64);

const COORDS_TO_NODE: phf::Map<u128, usize> = include_generated!("COORDS_TO_NODE.rs");

/// Returns the node index for the given coords, or None if there is not a
/// walkable tile there.
pub fn coords_to_node(coords: (i8, i8)) -> Option<usize> {
    let key = ((coords.0 as u128) << 64) | (coords.1 as u128);
    COORDS_TO_NODE.get(&key).copied()
}

pub const VALID_ACTIONS: [[bool; 5]; NUM_NODES] = include_generated!("VALID_ACTIONS.rs");
pub const ACTION_DISTRIBUTIONS: [[[f32; 5]; NUM_NODES]; NUM_NODES] =
    include_generated!("ACTION_DISTRIBUTIONS.rs");
pub const DISTANCE_MATRIX: [[u8; NUM_NODES]; NUM_NODES] = include_generated!("DISTANCE_MATRIX.rs");
pub const SUPER_PELLET_LOCS: [(i8, i8); 4] = include_generated!("SUPER_PELLET_LOCS.rs");
