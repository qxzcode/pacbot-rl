use std::{
    cell::RefCell,
    ffi::c_int,
    os::raw::{c_char, c_void},
    ptr,
};

use pyo3::{
    exceptions::{PyBufferError, PyIndexError, PyKeyError},
    ffi::{self, Py_ssize_t},
    prelude::*,
    AsPyPointer,
};
use static_assertions::assert_eq_size;

use super::GameState;
use crate::{ghost_agent::GhostAgent, variables::GridValue};

/// Wraps a reference to one of a GameState's ghosts.
#[pyclass]
struct GhostAgentWrapper {
    game_state: Py<GameState>,
    get_ghost: fn(&GameState) -> &RefCell<GhostAgent>,
}

#[pymethods]
impl GhostAgentWrapper {
    #[getter]
    fn pos(&self) -> GhostPosWrapper {
        GhostPosWrapper {
            game_state: self.game_state.clone(),
            get_ghost: self.get_ghost,
        }
    }

    fn clear_start_path(&mut self) -> PyResult<()> {
        Python::with_gil(|py| {
            let game_state = self.game_state.borrow(py);
            let mut ghost = (self.get_ghost)(&game_state).borrow_mut();
            ghost.start_path = &[];
            Ok(())
        })
    }
}

/// Wraps a reference to one of a ghost's positions.
#[pyclass]
struct GhostPosWrapper {
    game_state: Py<GameState>,
    get_ghost: fn(&GameState) -> &RefCell<GhostAgent>,
}

#[pymethods]
impl GhostPosWrapper {
    fn __getitem__(&self, item: &str) -> PyResult<(usize, usize)> {
        Python::with_gil(|py| {
            let game_state = self.game_state.borrow(py);
            let ghost = (self.get_ghost)(&game_state).borrow();
            match item {
                "current" => Ok(ghost.current_pos),
                "next" => Ok(ghost.next_pos),
                _ => Err(PyKeyError::new_err(item.to_owned())),
            }
        })
    }

    fn __setitem__(&self, item: &str, pos: (usize, usize)) -> PyResult<()> {
        Python::with_gil(|py| {
            let game_state = self.game_state.borrow(py);
            let mut ghost = (self.get_ghost)(&game_state).borrow_mut();
            match item {
                "current" => ghost.current_pos = pos,
                "next" => ghost.next_pos = pos,
                _ => return Err(PyKeyError::new_err(item.to_owned())),
            }
            Ok(())
        })
    }
}

pub(super) fn wrap_ghost_agent(
    game_state: Py<GameState>,
    get_ghost: fn(&GameState) -> &RefCell<GhostAgent>,
) -> impl IntoPy<Py<PyAny>> {
    GhostAgentWrapper {
        game_state,
        get_ghost,
    }
}

/// Wraps a reference to a GameState's grid.
#[pyclass]
struct GridWrapper {
    game_state: Py<GameState>,
}

#[pymethods]
impl GridWrapper {
    fn __getitem__(&self, index: usize) -> PyResult<GridRowWrapper> {
        Python::with_gil(|py| {
            let game_state = self.game_state.borrow(py);
            if index < game_state.grid.len() {
                Ok(GridRowWrapper {
                    game_state: self.game_state.clone(),
                    row: index,
                })
            } else {
                Err(PyIndexError::new_err("grid row index out of range"))
            }
        })
    }

    unsafe fn __getbuffer__(&self, view: *mut ffi::Py_buffer, flags: c_int) -> PyResult<()> {
        Python::with_gil(|py| {
            let game_state = self.game_state.borrow(py);
            let grid_shape_strides = Box::new([
                // shape:
                game_state.grid.len() as Py_ssize_t,
                game_state.grid[0].len() as Py_ssize_t,
                // strides:
                game_state.grid[0].len() as Py_ssize_t,
                1,
            ]);
            let grid_shape = &grid_shape_strides[..2];
            let grid_strides = &grid_shape_strides[2..];

            // adapted from https://github.com/PyO3/pyo3/blob/90d50da506d4090c6988b9b82225a21cf437e2e7/tests/test_buffer_protocol.rs#L162-L207

            if view.is_null() {
                return Err(PyBufferError::new_err("View is null"));
            }

            if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
                return Err(PyBufferError::new_err("Object is not writable"));
            }

            assert_eq_size!(GridValue, u8);

            *view = ffi::Py_buffer {
                obj: ffi::_Py_NewRef(self.game_state.as_ptr()),

                buf: game_state.grid.as_ptr() as *mut c_void,
                len: grid_shape.iter().product(),
                readonly: 1,
                itemsize: 1,

                format: if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                    "B\0".as_ptr() as *mut c_char
                } else {
                    ptr::null_mut()
                },

                ndim: 2,
                shape: if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
                    grid_shape.as_ptr() as *mut Py_ssize_t
                } else {
                    ptr::null_mut()
                },

                strides: if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                    grid_strides.as_ptr() as *mut Py_ssize_t
                } else {
                    ptr::null_mut()
                },

                suboffsets: ptr::null_mut(),
                internal: ptr::null_mut(),
            };

            Box::leak(grid_shape_strides); // don't drop the shape and strides buffer yet

            Ok(())
        })
    }

    unsafe fn __releasebuffer__(&self, view: *mut ffi::Py_buffer) {
        // release memory held by the shape and strides buffer
        drop(Box::from_raw((*view).shape as *mut [Py_ssize_t; 4]));
    }
}

/// Wraps a reference to a row of a GameState's grid.
#[pyclass]
struct GridRowWrapper {
    game_state: Py<GameState>,
    row: usize,
}

#[pymethods]
impl GridRowWrapper {
    fn __getitem__(&self, index: usize) -> PyResult<u8> {
        Python::with_gil(|py| {
            let game_state = self.game_state.borrow(py);
            if index < game_state.grid[self.row].len() {
                Ok(game_state.grid[self.row][index].into())
            } else {
                Err(PyIndexError::new_err("grid column index out of range"))
            }
        })
    }
}

pub(super) fn wrap_grid(game_state: Py<GameState>) -> impl IntoPy<Py<PyAny>> {
    GridWrapper { game_state }
}

/// Wraps a reference to a GameState's PacBot.
#[pyclass]
struct PacBotWrapper {
    game_state: Py<GameState>,
}

#[pymethods]
impl PacBotWrapper {
    #[getter]
    fn pos(&self) -> (usize, usize) {
        Python::with_gil(|py| {
            let game_state = self.game_state.borrow(py);
            game_state.pacbot.pos
        })
    }

    #[getter]
    fn direction(&self) -> u8 {
        Python::with_gil(|py| {
            let game_state = self.game_state.borrow(py);
            game_state.pacbot.direction.into()
        })
    }

    fn update(&self, position: (usize, usize)) {
        Python::with_gil(|py| {
            let mut game_state = self.game_state.borrow_mut(py);
            game_state.pacbot.update(position);
        })
    }
}

pub(super) fn wrap_pacbot(game_state: Py<GameState>) -> impl IntoPy<Py<PyAny>> {
    PacBotWrapper { game_state }
}
