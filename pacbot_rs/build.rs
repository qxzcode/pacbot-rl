//! Build script that precomputes various things from the game grid.

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::env;
use std::fmt::Debug;
use std::fs;
use std::io;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(non_camel_case_types)]
pub enum GridValue {
    /// Wall
    I = 1,
    /// Normal pellet
    o = 2,
    /// Empty space
    e = 3,
    /// Power pellet
    O = 4,
    /// Ghost chambers
    n = 5,
    /// Cherry position
    c = 6,
}
use GridValue::*;

pub const GRID: [[GridValue; 31]; 28] = include!("src/grid_data.txt");

fn output_count<P: AsRef<Path>>(cell_type: GridValue, out_path: P) -> io::Result<()> {
    let count = GRID.iter().flatten().filter(|v| **v == cell_type).count();
    fs::write(out_path, count.to_string())
}

fn output_array<T: npyz::Deserialize + Debug, P: AsRef<Path>>(
    in_path: &str,
    out_path: P,
) -> io::Result<()> {
    println!("cargo:rerun-if-changed={in_path}");
    let bytes = fs::read(in_path)?;
    let array = npyz::NpyFile::new(bytes.as_slice())?;

    let shape = array.shape().to_owned();
    let strides = array.strides().to_owned();
    let data = array.into_vec::<T>()?;
    let mut out_file = BufWriter::new(fs::File::create(out_path)?);

    fn output_subarray<T: Debug, W: Write>(
        out_file: &mut W,
        data: &[T],
        shape: &[u64],
        strides: &[u64],
    ) -> io::Result<()> {
        if shape.is_empty() {
            write!(out_file, "{:?}", data[0])
        } else {
            write!(out_file, "[")?;
            for i in 0..shape[0] {
                if i != 0 {
                    write!(out_file, ",")?;
                }
                let sub_data = &data[(i * strides[0]) as usize..];
                output_subarray(out_file, sub_data, &shape[1..], &strides[1..])?;
            }
            write!(out_file, "]")
        }
    }

    output_subarray(&mut out_file, &data, &shape, &strides)?;
    out_file.flush()
}

/// Performs a breadth-first search through the grid. For each node visited, calls
/// the given callback with the position and distance.
fn breadth_first_search(start: (usize, usize), mut callback: impl FnMut((usize, usize), usize)) {
    callback(start, 0);

    fn is_walkable(pos: (usize, usize)) -> bool {
        let tile = GRID[pos.0][pos.1];
        tile != GridValue::I && tile != GridValue::n
    }

    let mut queue = VecDeque::from([(start, 0)]);
    let mut visited = HashSet::from([start]);
    while let Some((cur_pos, cur_dist)) = queue.pop_front() {
        let neighbor_dist = cur_dist + 1;
        let (cx, cy) = cur_pos;
        for neighbor in [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)] {
            if is_walkable(neighbor) && visited.insert(neighbor) {
                callback(neighbor, neighbor_dist);
                queue.push_back((neighbor, neighbor_dist));
            }
        }
    }
}

fn compute_distance_matrix(node_coords: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let coords_to_node: HashMap<(usize, usize), usize> = node_coords
        .iter()
        .enumerate()
        .map(|(i, &pos)| (pos, i))
        .collect();
    let mut dists = vec![vec![0; node_coords.len()]; node_coords.len()];
    for (i, &start) in node_coords.iter().enumerate() {
        breadth_first_search(start, |pos, dist| dists[i][coords_to_node[&pos]] = dist);
    }
    dists
}

fn main() -> io::Result<()> {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);

    // pellet counts
    println!("cargo:rerun-if-changed=src/grid_data.txt");
    output_count(o, out_dir.join("GRID_PELLET_COUNT.txt"))?;
    output_count(O, out_dir.join("GRID_POWER_PELLET_COUNT.txt"))?;

    // node coordinates
    println!("cargo:rerun-if-changed=../computed_data/node_coords.json");
    let node_coords = include!("../computed_data/node_coords.json");
    let node_coords = node_coords.map(|pair| (pair[0], pair[1]));
    fs::write(out_dir.join("NODE_COORDS.rs"), format!("{node_coords:?}"))?;

    // map from coordinates to node index
    let mut coords_to_node = phf_codegen::Map::new();
    for (i, coords) in node_coords.iter().enumerate() {
        let key = ((coords.0 as u128) << 64) | (coords.1 as u128);
        coords_to_node.entry(key, &i.to_string());
    }
    fs::write(
        out_dir.join("COORDS_TO_NODE.rs"),
        coords_to_node.build().to_string(),
    )?;

    // node embeddings
    output_array::<f64, _>(
        "../computed_data/node_embeddings.npy",
        out_dir.join("NODE_EMBEDDINGS.rs"),
    )?;

    // valid actions
    let node_coords_set = HashSet::<_>::from_iter(node_coords);
    let valid_actions = node_coords.map(|(x, y)| {
        [
            true,
            node_coords_set.contains(&(x, y + 1)),
            node_coords_set.contains(&(x, y - 1)),
            node_coords_set.contains(&(x - 1, y)),
            node_coords_set.contains(&(x + 1, y)),
        ]
    });
    fs::write(
        out_dir.join("VALID_ACTIONS.rs"),
        format!("{valid_actions:?}"),
    )?;

    // action distributions
    output_array::<f32, _>(
        "../computed_data/action_distributions.npy",
        out_dir.join("ACTION_DISTRIBUTIONS.rs"),
    )?;

    // distance matrix
    let distance_matrix = compute_distance_matrix(&node_coords);
    fs::write(
        out_dir.join("DISTANCE_MATRIX.rs"),
        format!("{distance_matrix:?}"),
    )?;

    // super pellet locations
    let super_pellet_locs = node_coords
        .iter()
        .filter(|(x, y)| GRID[*x][*y] == GridValue::O)
        .collect::<Vec<_>>();
    fs::write(
        out_dir.join("SUPER_PELLET_LOCS.rs"),
        format!("{super_pellet_locs:?}"),
    )?;

    Ok(())
}
