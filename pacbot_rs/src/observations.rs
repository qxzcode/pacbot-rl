use ndarray::{concatenate, Array1, ArrayView1, Axis};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::{
    game_state::GameState,
    grid::{coords_to_node, ACTION_DISTRIBUTIONS, EMBED_DIM, NODE_EMBEDDINGS, VALID_ACTIONS},
    variables::GridValue,
};

#[pyfunction]
pub fn create_obs_semantic(game_state: &GameState) -> Py<PyArray1<f32>> {
    let pacman_node_index =
        coords_to_node(game_state.pacbot.pos).expect("PacBot is not on a walkable tile");
    let pacman_pos_embed = &NODE_EMBEDDINGS[pacman_node_index];

    let ghosts = [
        game_state.red.borrow(),
        game_state.pink.borrow(),
        game_state.orange.borrow(),
        game_state.blue.borrow(),
    ];

    let mut ghost_embed = Array1::zeros(EMBED_DIM);
    for ghost in &ghosts {
        let pos = ghost.current_pos;
        if let Some(node_index) = coords_to_node(pos) {
            ghost_embed += &ArrayView1::from(&NODE_EMBEDDINGS[node_index]);
        }
    }

    let (px, py) = game_state.pacbot.pos;
    let closest_ghost_dir = if let Some(closest_ghost_pos_index) = ghosts
        .iter()
        .map(|ghost| ghost.current_pos)
        .filter_map(|pos| coords_to_node(pos).map(|i| (pos, i)))
        .min_by_key(|&((gx, gy), _)| usize::abs_diff(px, gx) + usize::abs_diff(py, gy))
        .map(|(_, i)| i)
    {
        ACTION_DISTRIBUTIONS[pacman_node_index][closest_ghost_pos_index]
    } else {
        [0.0; 5]
    };

    let super_pellet_locs = [(1, 7), (1, 27), (26, 7), (26, 27)];
    let mut pellet_embed = Array1::zeros(EMBED_DIM);
    for pos in super_pellet_locs {
        let (x, y) = pos;
        if game_state.grid[x][y] == GridValue::O {
            let pos_embed = &NODE_EMBEDDINGS[coords_to_node(pos).unwrap()];
            pellet_embed += &ArrayView1::from(pos_embed);
        }
    }

    let is_frightened = game_state.is_frightened();
    let extra_info = [is_frightened as u8 as f32, pacman_node_index as f32];

    Python::with_gil(|py| {
        concatenate![
            Axis(0),
            pacman_pos_embed,
            ghost_embed,
            pellet_embed,
            closest_ghost_dir,
            VALID_ACTIONS[pacman_node_index].map(|b| b as u8 as f32),
            extra_info,
        ]
        .into_pyarray(py)
        .into()
    })
}
