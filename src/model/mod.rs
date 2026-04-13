use candle_core::{Device, Result};
use std::sync::mpsc;

use crate::{board::Board, model::queue::EvaluateRequest};

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

    let board_tensor = board.to_tensor(device)?;

    queue_tx.send((board_tensor, tx)).unwrap();

    let (mut prior, value) = rx.recv().unwrap();

    // TODO: Normalize prior over legal moves (maybe in the evaluation thread?)
    Ok((prior.remove(0), value))
}
