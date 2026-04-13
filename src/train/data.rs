use crate::board::{Board, Winner};

pub struct GameState {
    pub board: Board,
    pub distribution: Vec<f32>,
}

pub struct GameHistory {
    pub states: Vec<GameState>,
    pub winner: Winner,
}
