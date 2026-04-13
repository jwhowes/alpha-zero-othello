use core::slice::Iter;

use crate::board::{Board, Winner};

pub struct GameState {
    pub board: Board,
    pub distribution: Vec<f32>,
}

impl GameState {
    pub fn board(&self) -> &Board {
        &self.board
    }
}

pub struct GameHistory {
    pub states: Vec<GameState>,
    pub winner: Winner,
}

impl IntoIterator for GameHistory {
    type IntoIter = <Vec<GameState> as IntoIterator>::IntoIter;
    type Item = GameState;

    fn into_iter(self) -> Self::IntoIter {
        self.states.into_iter()
    }
}

impl GameHistory {
    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn winner(&self) -> Winner {
        self.winner
    }

    pub fn iter(&self) -> Iter<'_, GameState> {
        self.states.iter()
    }
}
