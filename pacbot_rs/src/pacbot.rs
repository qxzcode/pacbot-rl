use crate::variables::{Direction, PACBOT_STARTING_DIR, PACBOT_STARTING_POS};

#[derive(Clone)]
pub struct PacBot {
    pub pos: (usize, usize),
    pub direction: Direction,
}

impl PacBot {
    pub fn new() -> Self {
        Self {
            pos: PACBOT_STARTING_POS,
            direction: PACBOT_STARTING_DIR,
        }
    }

    pub fn respawn(&mut self) {
        self.pos = PACBOT_STARTING_POS;
        self.direction = PACBOT_STARTING_DIR;
    }

    pub fn update(&mut self, position: (usize, usize)) {
        if position.0 > self.pos.0 {
            self.direction = Direction::Right;
        } else if position.0 < self.pos.0 {
            self.direction = Direction::Left;
        } else if position.1 > self.pos.1 {
            self.direction = Direction::Up;
        } else if position.1 < self.pos.1 {
            self.direction = Direction::Down;
        }
        self.pos = position;
    }
}

impl Default for PacBot {
    fn default() -> Self {
        Self::new()
    }
}
