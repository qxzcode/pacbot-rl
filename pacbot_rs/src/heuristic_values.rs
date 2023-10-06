use std::collections::{HashSet, VecDeque};

use ordered_float::NotNan;
use pyo3::prelude::*;

use crate::{
    game_state::GameState,
    grid::{self, coords_to_node, DISTANCE_MATRIX, SUPER_PELLET_LOCS},
    variables::GridValue,
};

/// Performs a breadth-first search through the grid. Returns the distance from
/// the given start position to the closest position for which is_goal(pos) is
/// true, or returns None if no such position is reachable.
fn breadth_first_search(
    start: (usize, usize),
    mut is_goal: impl FnMut((usize, usize)) -> bool,
) -> Option<usize> {
    if is_goal(start) {
        return Some(0);
    }

    let mut queue = VecDeque::from([(start, 0)]);
    let mut visited = HashSet::from([start]);
    while let Some((cur_pos, cur_dist)) = queue.pop_front() {
        let neighbor_dist = cur_dist + 1;
        let (cx, cy) = cur_pos;
        for neighbor in [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)] {
            if grid::is_walkable(neighbor) && visited.insert(neighbor) {
                if is_goal(neighbor) {
                    return Some(neighbor_dist);
                }
                queue.push_back((neighbor, neighbor_dist));
            }
        }
    }
    None
}

const FEAR: u8 = 10;
const PELLET_WEIGHT: f32 = 0.65;
const SUPER_PELLET_WEIGHT: f32 = 10.0;
const GHOST_WEIGHT: f32 = 0.35;
const FRIGHTENED_GHOST_WEIGHT: f32 = 0.3 * GHOST_WEIGHT;

/// Computes the heuristic value from the hand-coded algorithm for the given
/// position in the game grid. Returns None if the given position is not walkable.
#[pyfunction]
pub fn get_heuristic_value(game_state: &GameState, pos: (usize, usize)) -> Option<f32> {
    let pos_node = coords_to_node(pos)?;

    let pellet_dist =
        breadth_first_search(pos, |(x, y)| game_state.grid[x][y] == GridValue::o).unwrap_or(0);
    let pellet_heuristic = pellet_dist as f32 * PELLET_WEIGHT;

    let super_pellet_dist = SUPER_PELLET_LOCS
        .iter()
        .filter(|&&(x, y)| game_state.grid[x][y] == GridValue::O)
        .map(|&pos| DISTANCE_MATRIX[pos_node][coords_to_node(pos).unwrap()])
        .min()
        .unwrap_or(0);
    let super_pellet_heuristic = super_pellet_dist as f32 * SUPER_PELLET_WEIGHT;

    // get the distance and frightened status of all (reachable) ghosts
    let ghost_dists = [
        game_state.red.borrow(),
        game_state.pink.borrow(),
        game_state.orange.borrow(),
        game_state.blue.borrow(),
    ]
    .into_iter()
    .filter_map(|ghost| {
        coords_to_node(ghost.current_pos)
            .map(|node| (DISTANCE_MATRIX[pos_node][node], ghost.is_frightened()))
    })
    .filter(|(dist, _)| *dist < FEAR);

    let ghost_heuristic: f32 = ghost_dists
        .into_iter()
        .map(|(dist, is_frightened)| {
            let weight = if is_frightened {
                -FRIGHTENED_GHOST_WEIGHT
            } else {
                GHOST_WEIGHT
            };
            let fear_minus_dist = FEAR - dist;
            (fear_minus_dist * fear_minus_dist) as f32 * weight
        })
        .sum();

    Some(pellet_heuristic + super_pellet_heuristic + ghost_heuristic)
}

/// Computes the heuristic values for each of the 5 actions for the given GameState.
/// Returns the values as well as the best action.
#[pyfunction]
pub fn get_action_heuristic_values(game_state: &GameState) -> ([Option<f32>; 5], u8) {
    let (px, py) = game_state.pacbot.pos;

    let values = [
        (px, py),
        (px, py + 1),
        (px, py - 1),
        (px - 1, py),
        (px + 1, py),
    ]
    .map(|pos| get_heuristic_value(game_state, pos));

    let best_action = values
        .iter()
        .enumerate()
        .filter_map(|(i, value)| value.map(|v| (i, v)))
        .min_by_key(|&(_, value)| NotNan::new(value).unwrap())
        .expect("at least one action should be valid")
        .0;

    (values, best_action.try_into().unwrap())
}
