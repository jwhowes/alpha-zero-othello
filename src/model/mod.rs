use candle_core::{Device, Result};
use std::sync::mpsc;

use crate::{
    board::{Board, Player, action::Action},
    model::queue::EvaluateRequest,
};

pub mod queue;
pub mod vit;

/*
 Evaluates a board position, returning the prior probabilities and non-subjective value
*/
pub fn evaluate_board(
    board: &Board,
    queue_tx: mpsc::Sender<EvaluateRequest>,
    device: &Device,
) -> Result<(Vec<f32>, f32)> {
    let (tx, rx) = oneshot::channel();

    queue_tx.send((board.to_tensor(device)?, tx)).unwrap();

    let (prior, value) = rx.recv().unwrap();

    let legal_actions = board.legal_actions();

    let mut legal_prior = Vec::with_capacity(legal_actions.len());

    for Action(x, y) in legal_actions.into_iter() {
        legal_prior.push(prior[y][x]);
    }

    let prior_sum: f32 = legal_prior.iter().sum();

    Ok((
        legal_prior.into_iter().map(|p| p / prior_sum).collect(),
        match board.player() {
            Player::Black => value,
            Player::White => -value,
        },
    ))
}
